/*
 * Metamorphosis Volume -- CUDA port for Jetson Nano (CUDA 10.2 / X11).
 * Original shader by dathor: https://www.shadertoy.com/view/NXySRV
 * Volume SDF technique by FabriceNeyret2:
 * https://www.shadertoy.com/view/NXGSRK
 * Ellipsoid distance approximation by Inigo Quilez.
 * Attribution does not replace the original authors' licensing terms.
 *
 * Port: Volume A -> cached shape fields + distance propagation + 3D texture;
 * Buffer A -> HDR scene/depth kernel; Image -> post-processing kernel.
 * Q/Escape: quit, Space: pause, R: restart, P: toggle post-processing.
 * --width N --height N --time S --output frame.ppm (headless single frame).
 * Windows builds support headless export for checking the CUDA kernels.
 */
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#ifndef _WIN32
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#endif

#define D __device__ __forceinline__
struct V2 {
    float x,y;
    D V2(float a=0):x(a),y(a){}
    D V2(float a,float b):x(a),y(b){}
};
struct V3 {
    float x,y,z;
    D V3(float a=0):x(a),y(a),z(a){}
    D V3(float a,float b,float c):x(a),y(b),z(c){}
};
struct V4 {
    float x,y,z,w;
    D V4(float a=0):x(a),y(a),z(a),w(a){}
    D V4(float a,float b,float c,float d):x(a),y(b),z(c),w(d){}
    D V4(V3 a,float b):x(a.x),y(a.y),z(a.z),w(b){}
    D V3 rgb()const{return V3(x,y,z);}
    D V2 xy()const{return V2(x,y);}
};
#define OPS(T,ARGS) \
 D T operator+(T a,T b){return T ARGS(+);} \
 D T operator-(T a,T b){return T ARGS(-);} \
 D T operator*(T a,T b){return T ARGS(*);} \
 D T operator/(T a,T b){return T ARGS(/);} \
 D T operator+(T a,float b){return a+T(b);} \
 D T operator+(float a,T b){return T(a)+b;} \
 D T operator-(T a,float b){return a-T(b);} \
 D T operator-(float a,T b){return T(a)-b;} \
 D T operator-(T a){return T(0)-a;} \
 D T operator*(T a,float b){return a*T(b);} \
 D T operator*(float a,T b){return T(a)*b;} \
 D T operator/(T a,float b){return a/T(b);} \
 D T operator/(float a,T b){return T(a)/b;} \
 D T& operator+=(T& a,T b){a=a+b;return a;} \
 D T& operator*=(T& a,float b){a=a*b;return a;}
#define A2(op) (a.x op b.x,a.y op b.y)
#define A3(op) (a.x op b.x,a.y op b.y,a.z op b.z)
#define A4(op) (a.x op b.x,a.y op b.y,a.z op b.z,a.w op b.w)
OPS(V2,A2)
OPS(V3,A3)
OPS(V4,A4)
#undef OPS
#undef A2
#undef A3
#undef A4
D float clamp(float x,float a=0,float b=1){return fminf(fmaxf(x,a),b);}
D V2 clamp(V2 v,float a=0,float b=1){return V2(clamp(v.x,a,b),clamp(v.y,a,b));}
D V3 clamp(V3 v,float a=0,float b=1){return V3(clamp(v.x,a,b),clamp(v.y,a,b),clamp(v.z,a,b));}
D float fract(float x){return x-floorf(x);}
D V3 fract(V3 x){return V3(fract(x.x),fract(x.y),fract(x.z));}
D float mod(float x,float y){return x-y*floorf(x/y);}
D float mix(float a,float b,float t){return a+(b-a)*t;}
D V3 mix(V3 a,V3 b,float t){return a+(b-a)*t;}
D float smooth(float a,float b,float x){float t=clamp((x-a)/(b-a));return t*t*(3-2*t);}
D V3 smooth(float a,float b,V3 x){return V3(smooth(a,b,x.x),smooth(a,b,x.y),smooth(a,b,x.z));}
D float dot(V2 a,V2 b){return a.x*b.x+a.y*b.y;}
D float dot(V3 a,V3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}
D float length(V2 a){return sqrtf(dot(a,a));}
D float length(V3 a){return sqrtf(dot(a,a));}
D V3 normalize(V3 a){return a/fmaxf(length(a),1e-12f);}
D V3 cross(V3 a,V3 b){return V3(a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x);}
D V3 reflect(V3 a,V3 n){return a-2*dot(a,n)*n;}
D V3 pow3(V3 a,float p){return V3(powf(a.x,p),powf(a.y,p),powf(a.z,p));}
D V2 xz(V3 p){return V2(p.x,p.z);}
D float morph(float t){float p=fract(t/40);return smooth(.20f,.40f,p)-smooth(.75f,.95f,p);}

// Volume A. Precompute the unmorphed fields once; neither depends on time.
const int VS=128, VN=VS*VS*VS;
D V3 gradient(V3 c){
    V3 v(dot(c,V3(467,581,389)),dot(c,V3(-394,-508,-316)),dot(c,V3(389,467,581)));
    return 2*fract(V3(sinf(v.x),sinf(v.y),sinf(v.z))*3758.54f)-1;
}
D float corner(V3 p,V3 cell,V3 off){V3 c=cell+off;return dot(gradient(c),p-c);}
D float perlin(V3 p){
    V3 c(floorf(p.x),floorf(p.y),floorf(p.z)),w=smooth(0,1,fract(p));
    float a=mix(corner(p,c,V3(0,0,0)),corner(p,c,V3(1,0,0)),w.x);
    float b=mix(corner(p,c,V3(0,1,0)),corner(p,c,V3(1,1,0)),w.x);
    float d=mix(corner(p,c,V3(0,0,1)),corner(p,c,V3(1,0,1)),w.x);
    float e=mix(corner(p,c,V3(0,1,1)),corner(p,c,V3(1,1,1)),w.x);
    return 2*fabsf(mix(mix(a,b,w.y),mix(d,e,w.y),w.z));
}
D float turbulence(V3 p){float v=0,s=1;for(int i=0;i<6;i++){v+=perlin(p)/s;p*=2;s*=2;}return v;}
D float hash11(float n){return fract(sinf(n*12.9898f)*43758.5453f);}
D float ellipsoid(V3 p,V3 r){float a=length(p/r),b=length(p/(r*r));return a*(a-1)/fmaxf(b,1e-4f);}
D float petals(V3 q,float count,float rot,float base,float y,float tilt,float len,float width,float thick,float bend,float cup,float seed){
    float radius=length(xz(q)),angle=atan2f(q.z,q.x)-rot,sector=6.2831853f/count;
    float index=roundf(angle/sector),local=angle-index*sector,d=1e9f;
    for(int j=-1;j<=1;j++){
        float a=local+j*sector,h=hash11(mod(index-j,count)+seed*31);
        float t=tilt+.12f*(h-.5f),l=len*(.9f+.2f*h);
        V3 p(radius*cosf(a)-base,q.y-y,radius*sinf(a));
        float c=cosf(t),s=sinf(t),u=p.x*c+p.y*s,v=-p.x*s+p.y*c;
        v-=bend*u*u+cup*p.z*p.z;
        d=fminf(d,ellipsoid(V3(u-.5f*l,v,p.z),V3(.5f*l,thick,width)));
    }return d;
}
D float lotus(V3 q){
    float d=petals(q,9,0,.10f,-.5f,.45f,.85f,.17f,.075f,.25f,2,1);
    d=fminf(d,petals(q,9,.349f,.10f,-.48f,.75f,.85f,.16f,.075f,.30f,2.5f,2));
    d=fminf(d,petals(q,8,0,.11f,-.46f,1,.80f,.15f,.07f,.30f,2.5f,3));
    d=fminf(d,petals(q,7,.449f,.12f,-.44f,1.20f,.65f,.13f,.07f,.30f,3,4));
    d=fminf(d,petals(q,6,0,.14f,-.42f,1.38f,.48f,.10f,.065f,.20f,3,5));
    V3 p=q-V3(0,-.22f,0);V2 w(length(xz(p))-.10f,fabsf(p.y)-.03f);
    float pod=fminf(fmaxf(w.x,w.y),0)+length(V2(fmaxf(w.x,0),fmaxf(w.y,0)))-.03f;
    return fminf(fminf(d,pod),length(q-V3(0,-.46f,0))-.12f);
}
__global__ void bake(float2* fields){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=VN)return;
    // Shadertoy mainVolume receives voxel centers.
    V3 q=(V3(i%VS,(i/VS)%VS,i/(VS*VS))+.5f)/64-1;
    float n=turbulence(q)*.5f-.274f;
    fields[i]=make_float2(length(q)-.7f+2*n,lotus(q)+.15f*n);
}
__global__ void propagate(float* dst,const float* prev,const float2* fields,float m){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=VN)return;
    float field=mix(fields[i].x,fields[i].y,m);
    if(field<=.1f){dst[i]=field;return;}
    int x=i%VS,y=(i/VS)%VS,z=i/(VS*VS);float best=1e9f,step=0;
    // Match Volume A: select the smallest neighbor value, then add its distance.
    for(int k=0;k<27;k++){
        if(k==13)continue;
        int dx=k%3-1,dy=(k/3)%3-1,dz=k/9-1;
        int xx=max(0,min(VS-1,x+dx)),yy=max(0,min(VS-1,y+dy)),zz=max(0,min(VS-1,z+dz));
        float v=prev[xx+VS*(yy+VS*zz)];
        if(v<best){best=v;step=sqrtf(float(dx*dx+dy*dy+dz*dz));}
    }dst[i]=best+step*(2.0f/VS);
}

// Buffer A: volume integration, animated camera, sky and water reflections.
struct Scene {cudaTextureObject_t volume;float t,m;int w,h;};
D float vol(const Scene& s,V3 p){return tex3D<float>(s.volume,.5f+.5f*p.x,.5f+.5f*p.y,.5f+.5f*p.z);}
D V3 light(){return V3(-.738f,.422f,.527f);}
D V2 box(V3 p,V3 d){
    // Avoid NaNs from rays parallel to a box face.
    d=V3(fabsf(d.x)<1e-8f?copysignf(1e-8f,d.x):d.x,fabsf(d.y)<1e-8f?copysignf(1e-8f,d.y):d.y,fabsf(d.z)<1e-8f?copysignf(1e-8f,d.z):d.z);
    V3 a=(-1-p)/d,b=(1-p)/d;
    return V2(fmaxf(fmaxf(fminf(a.x,b.x),fminf(a.y,b.y)),fminf(a.z,b.z)),fminf(fminf(fmaxf(a.x,b.x),fmaxf(a.y,b.y)),fmaxf(a.z,b.z)));
}
D V3 sky(V3 d){float y=fmaxf(d.y,0),a=fmaxf(dot(d,light()),0);return mix(V3(1,.62f,.42f),V3(.08f,.18f,.42f),powf(y,.45f))+V3(1,.6f,.3f)*(powf(a,8)*.35f+powf(a,400)*6);}
D V3 albedo(V3 p,float m){
    V3 a=mix(V3(.55f,.08f,.04f),V3(1,.55f,.2f),smooth(-.7f,.5f,p.y));
    float r=length(xz(p)),tip=clamp(r*.8f+(p.y+.5f)*.45f);
    V3 b=mix(V3(1,.90f,.86f),V3(.95f,.28f,.50f),smooth(.15f,.85f,tip));
    float pod=smooth(.18f,.12f,r)*smooth(-.34f,-.28f,p.y)*smooth(-.12f,-.20f,p.y);
    return mix(a,mix(b,V3(.85f,.70f,.15f),pod),m);
}
D V3 shade(const Scene& s,V3 p,V3 n,V3 d,float density){
    V3 a=albedo(p,s.m),l=light();float nl=dot(n,l),dif=fmaxf(nl,0),wrap=.5f+.5f*nl;wrap*=wrap;
    float ao=mix(.35f,1,clamp(vol(s,p+n*.06f)/.06f));
    float spec=powf(fmaxf(dot(n,normalize(l-d)),0),60),fre=powf(1-fmaxf(dot(n,-d),0),3);
    V3 amb=mix(V3(.04f,.09f,.10f),V3(.25f,.30f,.45f),.5f+.5f*n.y);
    float tr=powf(fmaxf(dot(d,l),0),3)*expf(density*15);
    V3 c=a*(V3(1,.85f,.7f)*(dif*1.2f+wrap*.3f)+amb)*ao;
    c+=spec*V3(1,.8f,.6f)*.5f*ao;c+=fre*sky(reflect(d,n))*.35f*ao;
    return c+a*mix(V3(1,.6f,.3f),V3(1,.65f,.7f),s.m)*tr*.8f;
}
D V4 integrate(const Scene& s,V3 p,V3 d,float l,float end,float& depth){
    V3 col(0);float T=1,r=1,dl=1.0f/128;int i=0;depth=1e9f;p+=l*d;
    for(;l<end&&r>.01f&&i<30;i++,p+=r*d,l+=r)r=vol(s,p);
    for(;l<end&&T>.01f&&i<1200;i++,p+=dl*d,l+=dl){
        float v=vol(s,p),a=smooth(0,-1.0f/128,v);
        if(a>0){
            V3 n=normalize(V3(vol(s,p+V3(.03f,0,0))-vol(s,p-V3(.03f,0,0)),vol(s,p+V3(0,.03f,0))-vol(s,p-V3(0,.03f,0)),vol(s,p+V3(0,0,.03f))-vol(s,p-V3(0,0,.03f))));
            col+=shade(s,p,n,d,v)*T*a*dl*9;T*=expf(-9*a*dl);
            if(depth>1e8f&&T<.5f)depth=l;
        }
    }return V4(col,T);
}
D V2 waterSlope(V2 q,float t,float dist){
    V2 swell(0),ripple(0);
    for(int i=0;i<3;i++){
        float f=float(i);V2 d(cosf(f*2.4f+.3f),sinf(f*2.4f+.3f));
        swell+=d*cosf(dot(q,d)*1.4f*powf(1.6f,f)+t*(.5f+.2f*f))*.035f/(1+.6f*f);
        d=V2(cosf(f*2.1f+1.2f),sinf(f*2.1f+1.2f));
        ripple+=d*cosf(dot(q,d)*9*powf(1.5f,f)+t*(1.6f+.5f*f))*.018f/(1+f);
    }
    float r=length(q),env=smooth(.5f,.9f,r)*expf(-fmaxf(r-.8f,0)*.9f);
    float rings=(cosf(r*16-t*2.2f)*.06f+cosf(r*27-t*3.1f)*.025f)*env;
    return swell/(1+.05f*dist)+ripple/(1+.25f*dist)+q/fmaxf(r,1e-3f)*rings/(1+.15f*dist);
}
D V4 key(int i){
    switch((i%8+8)%8){
        case 0:return V4(4.5f,1.2f,0,2.4f);case 1:return V4(3,.6f,0,2.6f);
        case 2:return V4(1.9f,-.3f,.1f,1.9f);case 3:return V4(1.5f,-.35f,.2f,1.7f);
        case 4:return V4(1.6f,.5f,.3f,2);case 5:return V4(1.5f,1.6f,0,1.6f);
        case 6:return V4(2.5f,1,0,2.4f);default:return V4(3.8f,.3f,0,3);
    }
}
D V4 camera(float u){int i=int(floorf(u));float f=fract(u);V4 a=key(i-1),b=key(i),c=key(i+1),d=key(i+2);return .5f*(2*b+(-a+c)*f+(2*a-5*b+4*c-d)*f*f+(-a+3*b-3*c+d)*f*f*f);}
__global__ void render(float4* image,Scene s){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=s.w||y>=s.h)return;
    float t=s.t;V2 uv=(2*V2(x+.5f,y+.5f)-V2(s.w,s.h))/float(s.h);
    V4 k=camera(t/5);float theta=.22f*t+.35f*sinf(.17f*t);
    V3 p(k.x*sinf(theta),fmaxf(k.y,-.42f),k.x*cosf(theta)),target(0,k.z-.1f,0);
    p+=.02f*V3(sinf(t*1.3f),sinf(t*1.7f+1),sinf(t*1.1f+2));
    target+=.015f*V3(sinf(t*1.9f+3),sinf(t*1.5f),sinf(t*1.2f+1));
    float roll=.03f*sinf(.4f*t);V3 ww=normalize(target-p),uu=normalize(cross(ww,V3(sinf(roll),cosf(roll),0))),vv=cross(uu,ww);
    V3 d=normalize(uv.x*uu+uv.y*vv+k.w*ww);
    if(x==0&&y==0){V3 a(dot(light(),uu),dot(light(),vv),dot(light(),ww));V2 v=V2(a.x,a.y)*k.w/fmaxf(a.z,1e-3f);v=.5f+.5f*v*V2(float(s.h)/s.w,1);image[0]=make_float4(v.x,v.y,smooth(0,.3f,a.z)*.99f,120);return;}
    V2 b=box(p,d);float tw=d.y<-1e-3f?(-.62f-p.y)/d.y:1e9f,vd=1e9f;V4 v(0,0,0,1);
    if(b.x<b.y&&b.y>0)v=integrate(s,p,d,fmaxf(b.x,0),fminf(b.y,tw),vd);
    V3 bg;float depth=120;
    if(tw<1e8f){
        V3 ph=p+tw*d;depth=fminf(tw*10,80);V2 g=waterSlope(xz(ph),t,tw);V3 nw=normalize(V3(-g.x,1,-g.y));
        V3 rf=reflect(d,nw);rf.y=fabsf(rf.y);V2 b2=box(ph,rf);V4 v2(0,0,0,1);float unused;
        if(b2.x<b2.y&&b2.y>0)v2=integrate(s,ph,rf,fmaxf(b2.x,0),b2.y,unused);
        V3 refl=v2.rgb()+v2.w*sky(rf);float f=.10f+.90f*powf(1-fmaxf(dot(nw,-d),0),4);
        V3 deep=V3(.004f,.022f,.035f)+V3(0,.02f,.02f)*(.5f+.5f*nw.x);
        bg=mix(sky(normalize(V3(d.x,0,d.z))),mix(deep,refl,f),expf(-.03f*tw));
    }else bg=sky(d);
    if(vd<1e8f)depth=vd*10;V3 c=v.rgb()+v.w*bg;image[x+s.w*y]=make_float4(c.x,c.y,c.z,depth);
}

// Image pass. Match Shadertoy's nearest/clamp Buffer A sampler.
struct Post {const float4* image;int w,h;float t;};
D V4 sample(const Post& s,V2 uv){int x=min(s.w-1,max(0,int(floorf(uv.x*s.w)))),y=min(s.h-1,max(0,int(floorf(uv.y*s.h))));float4 v=s.image[x+y*s.w];return V4(v.x,v.y,v.z,v.w);}
D float luma(V3 c){return dot(c,V3(.299f,.587f,.114f));}
D float hash12(V2 p){V3 a=fract(V3(p.x,p.y,p.x)*.1031f);a+=V3(dot(a,V3(a.y,a.z,a.x)+33.33f));return fract((a.x+a.y)*a.z);}
D float noise2(V2 p){V2 i(floorf(p.x),floorf(p.y)),f(fract(p.x),fract(p.y));f=f*f*(3-2*f);return mix(mix(hash12(i),hash12(i+V2(1,0)),f.x),mix(hash12(i+V2(0,1)),hash12(i+V2(1,1)),f.x),f.y);}
D float fbm2(V2 p){float s=0,a=.5f;for(int i=0;i<4;i++){s+=a*noise2(p);p=p*2.02f+V2(3.1f,1.7f);a*=.5f;}return s;}
D V3 karis(const Post& s,V2 uv,V2 ts){
    V3 a=sample(s,uv).rgb();float w=2/(1+luma(a));V3 sum=a*w;
    for(int i=0;i<4;i++){V3 c=sample(s,uv+V2((i&1)?.75f:-.75f,(i&2)?.75f:-.75f)*ts).rgb();float k=1/(1+luma(c));sum+=c*k;w+=k;}return sum/w;
}
D float coc(float d){float c=smooth(20,70,fabsf(d-25))*.6f;c*=1-smooth(85,105,d);return fmaxf(c,smooth(12,0,d));}
D V3 bloom(const Post& s,V2 uv,float dith,float bar,float aspect){
    V3 sum(0);for(int i=0;i<24;i++){
        float r=sqrtf((i+.5f)/24)*.02f,a=i*2.39996323f+dith*6.28318f;
        V2 off(cosf(a)*r/aspect,sinf(a)*r);V4 v=sample(s,V2(clamp(uv.x+off.x),clamp(uv.y+off.y,bar,1-bar)));
        float f=smooth(30,60,v.w)*(1-smooth(105,115,v.w)),l=luma(v.rgb());sum+=v.rgb()*(f*fmaxf(l-.25f,0)/(1+l));
    }return sum/24*1.5f*V3(1,.72f,.45f);
}
D V3 atmosphere(const Post& s,V2 uv,float dith,float aspect,float depth){
    float isSky=smooth(90,110,depth);V2 q=V2(uv.x*aspect,uv.y)*2;
    float n=fbm2(q+V2(s.t*.012f,-s.t*.008f)),n2=fbm2(q*2.2f+V2(-s.t*.02f,s.t*.01f)+5.2f);
    float wisps=smooth(.35f,.75f,n*.65f+n2*.35f),fog=smooth(30,52,depth)*(1-isSky);
    V3 haze=V3(.10f,.18f,.30f)*.30f*fog*(.4f+.6f*wisps),halo(0);
    if(isSky>.01f){V3 sum(0);float w=0;for(int i=0;i<16;i++){
        float r=sqrtf((i+.5f)/16)*.07f,a=i*2.39996323f+dith*6.28318f;V4 v=sample(s,clamp(uv+V2(cosf(a)*r/aspect,sinf(a)*r)));
        float wg=1-.7f*(r/.07f);sum+=v.rgb()*(1-smooth(90,110,v.w))*wg/(1+luma(v.rgb()));w+=wg;
    }halo=sum/w*.5f*V3(.55f,.75f,1)*(.5f+.8f*wisps);}return haze+halo*isSky;
}
D V3 rays(const Post& s,V2 uv,float dith,float aspect){
    float4 sun=s.image[0];if(sun.z<=0)return V3(0);V2 delta=(V2(sun.x,sun.y)-uv)*.95f/48,pos=uv+delta*dith;
    float illum=1,sum=0;for(int i=0;i<48;i++){
        V2 dv=(pos-V2(sun.x,sun.y))*V2(aspect,1);float d2=dot(dv,dv),glow=expf(-d2/.012f)+.15f*expf(-sqrtf(d2)*3),open=1;
        if(pos.x>=0&&pos.x<=1&&pos.y>=0&&pos.y<=1)open=smooth(90,110,sample(s,pos).w);
        sum+=glow*open*illum;illum*=.965f;pos+=delta;
    }return V3(1,.70f,.42f)*powf(sum/48,.8f)*1.1f*sun.z;
}
__global__ void postprocess(uchar4* out,Post s,bool effects){
    int x=blockIdx.x*blockDim.x+threadIdx.x,y=blockIdx.y*blockDim.y+threadIdx.y;if(x>=s.w||y>=s.h)return;
    V2 fc(x+.5f,y+.5f),ts(1.0f/s.w,1.0f/s.h),uv=fc*ts;float aspect=float(s.w)/s.h,bar=fmaxf(0,(1-aspect/2.35f)*.5f);
    // HDR buffer and Shadertoy coordinates are bottom-up; display output is top-down.
    int idx=x+(s.h-1-y)*s.w;
    if(uv.y<bar||uv.y>1-bar){out[idx]=make_uchar4(0,0,0,255);return;}
    V2 v=uv-.5f;float dith=fract(sinf(dot(fc,V2(12.9898f,78.233f)))*43758.5453f);V3 color=sample(s,uv).rgb();
    if(effects){
        V2 off=v*.003f*dot(v,v);V3 a=karis(s,clamp(uv-off),ts),b=karis(s,uv,ts),c=karis(s,clamp(uv+off),ts),sharp(a.x,b.y,c.z);color=sharp;
        float depth=sample(s,uv).w,co=coc(depth);
        if(co>.01f){V3 sum(0);float wsum=0;for(int i=0;i<16;i++){
            float r=sqrtf(i+.5f)/4,theta=i*2.39996323f+dith*6.28318f;V2 offset=V2(cosf(theta)/aspect,sinf(theta))*r*co*.0055f;
            V4 tap=sample(s,clamp(uv+offset));float w=1;if(tap.w<depth)w=clamp(coc(tap.w)/fmaxf(co,1e-3f));w/=1+luma(tap.rgb());sum+=tap.rgb()*w;wsum+=w;
        }color=mix(sharp,wsum>.001f?sum/wsum:sharp,smooth(.01f,.1f,co));}
        color+=bloom(s,uv,dith,bar,aspect);color+=atmosphere(s,uv,dith,aspect,depth);color+=rays(s,uv,dith,aspect);
    }
    color*=.9f;color=color/(1+color);color=pow3(clamp(color,0,1e9f),1/2.2f);float lum=luma(color);
    color=color*mix(V3(.85f,.97f,1.10f),V3(1.10f,1,.88f),smooth(.1f,.8f,lum));color=mix(V3(lum),color,1.15f);color=mix(color,smooth(0,1,color),.5f);
    color*=mix(.72f,1,smooth(1.4f,.35f,length(v)));color+=V3((dith-.5f)*(1.5f/255));
    float lt=mod(s.t,40),fade=(s.t<1.5f?1:smooth(0,1.5f,lt))*smooth(0,1.5f,40-lt);color=clamp(color*fade);
    out[idx]=make_uchar4((unsigned char)(255*color.z),(unsigned char)(255*color.y),(unsigned char)(255*color.x),255);
}

static void check(cudaError_t e,const char* call,int line){if(e!=cudaSuccess){fprintf(stderr,"CUDA line %d: %s: %s\n",line,call,cudaGetErrorString(e));exit(1);}}
#define CUDA(call) check((call),#call,__LINE__)
#ifndef _WIN32
static double seconds(){using namespace std::chrono;return duration<double>(steady_clock::now().time_since_epoch()).count();}
#endif
struct Renderer {
    float2* fields;float *prev,*next;float4* hdr;uchar4* pixels;cudaArray_t array;cudaTextureObject_t texture;int w,h;
    Renderer(int width,int height):w(width),h(height){
        CUDA(cudaMalloc(&fields,VN*sizeof(float2)));CUDA(cudaMalloc(&prev,VN*sizeof(float)));CUDA(cudaMalloc(&next,VN*sizeof(float)));
        CUDA(cudaMemset(prev,0,VN*sizeof(float)));CUDA(cudaMalloc(&hdr,size_t(w)*h*sizeof(float4)));CUDA(cudaMalloc(&pixels,size_t(w)*h*sizeof(uchar4)));
        cudaChannelFormatDesc desc=cudaCreateChannelDesc<float>();CUDA(cudaMalloc3DArray(&array,&desc,make_cudaExtent(VS,VS,VS)));
        cudaResourceDesc res={};res.resType=cudaResourceTypeArray;res.res.array.array=array;cudaTextureDesc tex={};
        tex.normalizedCoords=1;tex.filterMode=cudaFilterModeLinear;tex.readMode=cudaReadModeElementType;
        tex.addressMode[0]=tex.addressMode[1]=tex.addressMode[2]=cudaAddressModeClamp;CUDA(cudaCreateTextureObject(&texture,&res,&tex,NULL));
        puts("Baking the two 128^3 shape fields (one-time startup cost)...");bake<<<(VN+255)/256,256>>>(fields);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    }
    void update(float t){
        float p=t/40-floorf(t/40); // Host version of morph(), using the same smoothstep.
        float a=fminf(fmaxf((p-.20f)/.20f,0),1),b=fminf(fmaxf((p-.75f)/.20f,0),1);float m=a*a*(3-2*a)-b*b*(3-2*b);
        propagate<<<(VN+255)/256,256>>>(next,prev,fields,m);CUDA(cudaGetLastError());float* tmp=prev;prev=next;next=tmp;
    }
    void frame(float t,bool effects){
        update(t);cudaMemcpy3DParms cp={};cp.srcPtr=make_cudaPitchedPtr(prev,VS*sizeof(float),VS,VS);cp.dstArray=array;cp.extent=make_cudaExtent(VS,VS,VS);cp.kind=cudaMemcpyDeviceToDevice;CUDA(cudaMemcpy3D(&cp));
        dim3 block(16,8),grid((w+15)/16,(h+7)/8);Scene s={texture,t,0,w,h};
        float p=t/40-floorf(t/40),a=fminf(fmaxf((p-.20f)/.20f,0),1),b=fminf(fmaxf((p-.75f)/.20f,0),1);s.m=a*a*(3-2*a)-b*b*(3-2*b);
        render<<<grid,block>>>(hdr,s);CUDA(cudaGetLastError());Post post={hdr,w,h,t};postprocess<<<grid,block>>>(pixels,post,effects);CUDA(cudaGetLastError());
    }
    ~Renderer(){cudaDestroyTextureObject(texture);cudaFreeArray(array);cudaFree(fields);cudaFree(prev);cudaFree(next);cudaFree(hdr);cudaFree(pixels);}
};
int main(int argc,char** argv){
    int w=480,h=270;float time=0;bool effects=true;const char* output=NULL;
    for(int i=1;i<argc;i++){
        if(!strcmp(argv[i],"--no-post")){effects=false;continue;}
        if(!strcmp(argv[i],"--help")){puts("cuda_metamorphosis [--width N] [--height N] [--time S] [--no-post] [--output file.ppm]\nKeys: Q/Escape quit, Space pause, R restart, P toggle post-processing");return 0;}
        if(i+1>=argc){fprintf(stderr,"Missing value for %s\n",argv[i]);return 1;}
        if(!strcmp(argv[i],"--width"))w=atoi(argv[++i]);else if(!strcmp(argv[i],"--height"))h=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--time"))time=(float)atof(argv[++i]);else if(!strcmp(argv[i],"--output"))output=argv[++i];
        else{fprintf(stderr,"Unknown option: %s\n",argv[i]);return 1;}
    }
    if(w<16||h<16||w>4096||h>4096||!std::isfinite(time)||time<0){fputs("Width/height must be 16..4096; time must be finite and nonnegative.\n",stderr);return 1;}
#ifdef _WIN32
    if(!output){fputs("Use --output frame.ppm on Windows; interactive display uses X11 on Jetson.\n",stderr);return 1;}
#endif
#ifndef _WIN32
    Display* display=NULL;Window window=0;GC gc=0;XImage* image=NULL;Atom closeAtom=0;
    if(!output){
        display=XOpenDisplay(NULL);if(!display){fputs("Cannot open X11 display. Run on the Jetson desktop or use --output frame.ppm.\n",stderr);return 1;}
        int screen=DefaultScreen(display);Visual* visual=DefaultVisual(display,screen);int depth=DefaultDepth(display,screen);
        if(visual->red_mask!=0xff0000||visual->green_mask!=0xff00||visual->blue_mask!=0xff){fputs("Requires an RGB TrueColor X11 visual.\n",stderr);XCloseDisplay(display);return 1;}
        window=XCreateSimpleWindow(display,RootWindow(display,screen),0,0,w,h,0,0,0);XSelectInput(display,window,ExposureMask|KeyPressMask);
        closeAtom=XInternAtom(display,"WM_DELETE_WINDOW",False);XSetWMProtocols(display,window,&closeAtom,1);
        XSizeHints hints={};hints.flags=PMinSize|PMaxSize;hints.min_width=hints.max_width=w;hints.min_height=hints.max_height=h;XSetWMNormalHints(display,window,&hints);
        XStoreName(display,window,"CUDA Metamorphosis Volume");XMapWindow(display,window);gc=XCreateGC(display,window,0,NULL);
        image=XCreateImage(display,visual,depth,ZPixmap,0,NULL,w,h,32,0);
        if(!image||image->bits_per_pixel!=32||image->byte_order!=LSBFirst){fputs("Requires a 32-bit little-endian XImage.\n",stderr);if(image)XDestroyImage(image);XFreeGC(display,gc);XDestroyWindow(display,window);XCloseDisplay(display);return 1;}
        image->data=(char*)calloc(image->bytes_per_line,h);if(!image->data){fputs("Cannot allocate XImage.\n",stderr);XDestroyImage(image);XFreeGC(display,gc);XDestroyWindow(display,window);XCloseDisplay(display);return 1;}
    }
#endif
    cudaDeviceProp prop;CUDA(cudaGetDeviceProperties(&prop,0));printf("Metamorphosis Volume by dathor | GPU: %s | %dx%d\n",prop.name,w,h);
    Renderer renderer(w,h);std::vector<uchar4> host(size_t(w)*h);
    // Warm up distance propagation at the requested morph. The original pass
    // converges over successive frames; this avoids a blank first screenshot.
    for(int i=0;i<128;i++)renderer.update(time);CUDA(cudaDeviceSynchronize());
    if(output){
        renderer.frame(time,effects);CUDA(cudaMemcpy(host.data(),renderer.pixels,host.size()*sizeof(uchar4),cudaMemcpyDeviceToHost));
        FILE* f=fopen(output,"wb");if(!f){perror(output);return 1;}bool ok=fprintf(f,"P6\n%d %d\n255\n",w,h)>0;
        for(size_t i=0;i<host.size()&&ok;i++){unsigned char rgb[]={host[i].z,host[i].y,host[i].x};ok=fwrite(rgb,1,3,f)==3;}
        if(fclose(f)!=0)ok=false;if(!ok){fputs("Failed to write image.\n",stderr);return 1;}printf("Saved %s at t=%.3fs\n",output,time);return 0;
    }
#ifndef _WIN32
    bool running=true,paused=false;double last=seconds(),fpsStart=last;int frames=0;
    puts("Q/Escape quit | Space pause | R restart | P post-processing");
    while(running){
        while(XPending(display)){
            XEvent e;XNextEvent(display,&e);if(e.type==ClientMessage&&(Atom)e.xclient.data.l[0]==closeAtom)running=false;
            if(e.type==KeyPress){KeySym k=XLookupKeysym(&e.xkey,0);if(k==XK_q||k==XK_Q||k==XK_Escape)running=false;else if(k==XK_space)paused=!paused;else if(k==XK_r||k==XK_R){time=0;CUDA(cudaMemset(renderer.prev,0,VN*sizeof(float)));}else if(k==XK_p||k==XK_P)effects=!effects;}
        }
        if(!running)break;double now=seconds();if(!paused)time+=float(now-last);last=now;
        renderer.frame(time,effects);CUDA(cudaMemcpy(host.data(),renderer.pixels,host.size()*sizeof(uchar4),cudaMemcpyDeviceToHost));
        for(int y=0;y<h;y++)memcpy(image->data+y*image->bytes_per_line,host.data()+size_t(y)*w,size_t(w)*sizeof(uchar4));
        XPutImage(display,window,gc,image,0,0,0,0,w,h);XFlush(display);frames++;now=seconds();
        if(now-fpsStart>=1){char title[180];snprintf(title,sizeof(title),"Metamorphosis Volume | %.1f FPS | t=%.1fs | post %s",frames/(now-fpsStart),time,effects?"on":"off");XStoreName(display,window,title);frames=0;fpsStart=now;}
    }
    XDestroyImage(image);XFreeGC(display,gc);XDestroyWindow(display,window);XCloseDisplay(display);
#endif
    return 0;
}
