# CUDA Molecule Visualization

A real-time 3D molecular visualization program using CUDA ray tracing. Renders ball-and-stick models of molecules with accurate CPK (Corey-Pauling-Koltun) coloring and interactive rotation.

## Features

- **Real-time ray tracing** on NVIDIA GPUs
- **115 molecules** covering organic chemistry, biochemistry, vitamins, and common compounds
- **Ball-and-stick representation** with proper atomic radii
- **CPK coloring scheme**:
  - Hydrogen (H) - White
  - Carbon (C) - Dark Gray
  - Nitrogen (N) - Blue
  - Oxygen (O) - Red
  - Phosphorus (P) - Orange
  - Sulfur (S) - Yellow
  - Chlorine (Cl) - Green
  - Bromine (Br) - Dark Red
  - Fluorine (F) - Light Green
  - Iodine (I) - Purple
- **Bond visualization** with single, double, and triple bond rendering
- **Auto-rotation** with manual control

## Controls

| Key | Action |
|-----|--------|
| A / D | Previous / Next molecule |
| R | Random molecule |
| ← → | Rotate left / right |
| ↑ ↓ | Rotate up / down |
| W / S | Zoom in / out |
| Space | Pause / resume auto-rotation |
| Q / Esc | Quit |

---

## Molecule Library (115 Molecules)

### Simple Inorganic Molecules

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 1 | Water | H₂O | Essential solvent, bent geometry |
| 9 | Ammonia | NH₃ | Trigonal pyramidal, important base |
| 10 | Carbon Dioxide | CO₂ | Linear, greenhouse gas |
| 26 | Nitric Oxide | NO | Free radical, signaling molecule |
| 27 | Hydrogen Peroxide | H₂O₂ | Oxidizer, antiseptic |
| 33 | Oxygen | O₂ | Diatomic, essential for respiration |
| 34 | Nitrogen | N₂ | Diatomic, 78% of atmosphere |
| 35 | Hydrogen | H₂ | Simplest molecule, fuel |
| 36 | Ozone | O₃ | Bent triatomic, UV absorber |
| 37 | Carbon Monoxide | CO | Toxic gas, triple bond |
| 63 | Hydrogen Sulfide | H₂S | Rotten egg smell, bent geometry |

### Inorganic Acids

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 28 | Sulfuric Acid | H₂SO₄ | Strong acid, industrial chemical |
| 29 | Phosphoric Acid | H₃PO₄ | Food additive, fertilizers |
| 40 | Hydrogen Chloride | HCl | Strong acid when dissolved |
| 41 | Nitric Acid | HNO₃ | Strong oxidizing acid |
| 38 | Nitrous Oxide | N₂O | Laughing gas, anesthetic |
| 39 | Sulfur Dioxide | SO₂ | Preservative, volcanic gas |

### Alkanes & Simple Hydrocarbons

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 2 | Methane | CH₄ | Simplest alkane, natural gas |
| 43 | Ethane | C₂H₆ | Two-carbon alkane |
| 14 | Propane | C₃H₈ | Fuel gas, LPG component |
| 15 | Butane | C₄H₁₀ | Lighter fuel, four carbons |
| 16 | Cyclohexane | C₆H₁₂ | Chair conformation, solvent |

### Alkenes & Alkynes

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 62 | Ethylene | C₂H₄ | Simplest alkene, plant hormone |
| 44 | Propene | C₃H₆ | Propylene, polymer precursor |
| 32 | Acetylene | C₂H₂ | Triple bond, welding fuel |
| 82 | Phenylacetylene | C₈H₆ | Aromatic with triple bond |

### Aromatic Compounds

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 3 | Benzene | C₆H₆ | Aromatic ring, solvent |
| 30 | Toluene | C₇H₈ | Methylbenzene, solvent |
| 31 | Phenol | C₆H₅OH | Carbolic acid, antiseptic |
| 17 | Naphthalene | C₁₀H₈ | Mothballs, fused rings |
| 55 | Chlorobenzene | C₆H₅Cl | Halogenated aromatic |
| 56 | Nitrobenzene | C₆H₅NO₂ | Almond odor, precursor |
| 57 | Aniline | C₆H₅NH₂ | Aromatic amine, dye precursor |
| 58 | Styrene | C₈H₈ | Vinyl benzene, polystyrene monomer |
| 59 | Benzoic Acid | C₇H₆O₂ | Preservative, simplest aromatic acid |
| 78 | Benzaldehyde | C₇H₆O | Almond extract, aromatic aldehyde |
| 79 | Bromobenzene | C₆H₅Br | Halogenated aromatic |
| 80 | p-Xylene | C₈H₁₀ | Dimethylbenzene, solvent |
| 81 | Anisole | C₇H₈O | Methoxybenzene, fragrance |

### Alcohols

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 4 | Ethanol | C₂H₅OH | Drinking alcohol, solvent |
| 42 | Methanol | CH₃OH | Wood alcohol, toxic |
| 45 | Isopropanol | C₃H₇OH | Rubbing alcohol |
| 65 | tert-Butanol | C₄H₁₀O | Tertiary alcohol, solvent |
| 66 | 1-Butanol | C₄H₁₀O | Primary alcohol, solvent |
| 46 | Ethylene Glycol | C₂H₆O₂ | Antifreeze, diol |
| 47 | Glycerol | C₃H₈O₃ | Triol, moisturizer |

### Ethers

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 67 | Diethyl Ether | C₄H₁₀O | Classic anesthetic, solvent |
| 68 | MTBE | C₅H₁₂O | Methyl tert-butyl ether, fuel additive |
| 69 | THF | C₄H₈O | Tetrahydrofuran, polar aprotic solvent |
| 70 | 1,4-Dioxane | C₄H₈O₂ | Cyclic ether, stabilizer |

### Aldehydes & Ketones

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 11 | Formaldehyde | CH₂O | Simplest aldehyde, preservative |
| 48 | Acetaldehyde | C₂H₄O | Ethanal, metabolite |
| 12 | Acetone | C₃H₆O | Simplest ketone, nail polish remover |

### Carboxylic Acids

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 49 | Formic Acid | CH₂O₂ | Ant venom, simplest carboxylic acid |
| 13 | Acetic Acid | C₂H₄O₂ | Vinegar |
| 75 | Propionic Acid | C₃H₆O₂ | Food preservative |
| 76 | Butyric Acid | C₄H₈O₂ | Rancid butter smell |
| 50 | Lactic Acid | C₃H₆O₃ | Muscle fatigue, dairy |
| 77 | Succinic Acid | C₄H₆O₄ | Dicarboxylic acid, Krebs cycle |

### Esters & Anhydrides

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 51 | Ethyl Acetate | C₄H₈O₂ | Fruity solvent, nail polish |
| 73 | Methyl Acetate | C₃H₆O₂ | Fast-evaporating solvent |
| 74 | Acetic Anhydride | C₄H₆O₃ | Acetylation reagent |

### Halogenated Compounds

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 54 | Dichloromethane | CH₂Cl₂ | DCM, paint stripper |
| 64 | Chloroform | CHCl₃ | Historic anesthetic |
| 72 | Carbon Tetrachloride | CCl₄ | Fully halogenated, toxic |

### Polar Aprotic Solvents

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 52 | Acetonitrile | C₂H₃N | HPLC solvent |
| 53 | DMSO | C₂H₆OS | Dimethyl sulfoxide, penetrant |
| 71 | DMF | C₃H₇NO | Dimethylformamide, polar solvent |

### Nitrogen Compounds

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 18 | Urea | CH₄N₂O | Metabolic waste, fertilizer |

### Nucleobases

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 6 | Adenine | C₅H₅N₅ | Purine base, DNA/RNA (A) |
| 23 | Guanine | C₅H₅N₅O | Purine base, DNA/RNA (G) |
| 21 | Thymine | C₅H₆N₂O₂ | Pyrimidine base, DNA only (T) |
| 22 | Cytosine | C₄H₅N₃O | Pyrimidine base, DNA/RNA (C) |

### Sugars

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 7 | Glucose | C₆H₁₂O₆ | Blood sugar, energy source |
| 83 | Fructose | C₆H₁₂O₆ | Fruit sugar, sweetest natural sugar |
| 84 | Ribose | C₅H₁₀O₅ | RNA sugar backbone |
| 85 | Deoxyribose | C₅H₁₀O₄ | DNA sugar backbone |

### Amino Acids - Nonpolar

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 19 | Glycine | C₂H₅NO₂ | Simplest amino acid |
| 20 | Alanine | C₃H₇NO₂ | Small nonpolar |
| 60 | Valine | C₅H₁₁NO₂ | Branched-chain, essential |
| 61 | Leucine | C₆H₁₃NO₂ | Branched-chain, essential |
| 86 | Isoleucine | C₆H₁₃NO₂ | Branched-chain, essential |
| 96 | Proline | C₅H₉NO₂ | Cyclic, helix breaker |
| 98 | Methionine | C₅H₁₁NO₂S | Sulfur-containing, start codon |
| 91 | Phenylalanine | C₉H₁₁NO₂ | Aromatic, essential |
| 95 | Tryptophan | C₁₁H₁₂N₂O₂ | Aromatic, largest amino acid |

### Amino Acids - Polar Uncharged

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 87 | Serine | C₃H₇NO₃ | Hydroxyl group, phosphorylation site |
| 88 | Threonine | C₄H₉NO₃ | Hydroxyl group, essential |
| 97 | Cysteine | C₃H₇NO₂S | Thiol group, disulfide bonds |
| 92 | Tyrosine | C₉H₁₁NO₃ | Phenolic hydroxyl |

### Amino Acids - Acidic

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 89 | Aspartic Acid | C₄H₇NO₄ | Negatively charged at pH 7 |
| 90 | Glutamic Acid | C₅H₉NO₄ | MSG precursor, neurotransmitter |

### Amino Acids - Basic

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 91 | Lysine | C₆H₁₄N₂O₂ | Positively charged, essential |
| 100 | Arginine | C₆H₁₄N₄O₂ | Guanidinium group, NO precursor |
| 94 | Histidine | C₆H₉N₃O₂ | Imidazole ring, pH sensitive |

### Neurotransmitters

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 24 | Dopamine | C₈H₁₁NO₂ | Reward, motivation |
| 25 | Serotonin | C₁₀H₁₂N₂O | Mood, sleep regulation |

### Pharmaceuticals & Bioactive

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 5 | Caffeine | C₈H₁₀N₄O₂ | Stimulant, adenosine antagonist |
| 8 | Aspirin | C₉H₈O₄ | Acetylsalicylic acid, pain reliever |

### Metabolic Intermediates

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 99 | Pyruvate | C₃H₄O₃ | Glycolysis end product, Krebs cycle entry |

### Vitamins

#### Water-Soluble Vitamins

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 101 | Vitamin C | C₆H₈O₆ | L-Ascorbic acid, antioxidant, collagen synthesis |
| 102 | Vitamin B₁ | C₁₂H₁₇N₄OS⁺ | Thiamine, carbohydrate metabolism |
| 103 | Vitamin B₂ | C₁₇H₂₀N₄O₆ | Riboflavin, FAD/FMN precursor |
| 104 | Vitamin B₃ | C₆H₅NO₂ | Niacin (Nicotinic acid), NAD precursor |
| 105 | Vitamin B₅ | C₉H₁₇NO₅ | Pantothenic acid, CoA component |
| 106 | Vitamin B₆ | C₈H₁₁NO₃ | Pyridoxine, amino acid metabolism |
| 107 | Vitamin B₇ | C₁₀H₁₆N₂O₃S | Biotin, carboxylation reactions |
| 108 | Vitamin B₉ | C₁₉H₁₉N₇O₆ | Folic acid, DNA synthesis |
| 114 | Nicotinamide | C₆H₆N₂O | Niacinamide, NAD⁺ precursor |

#### Fat-Soluble Vitamins

| # | Name | Formula | Description |
|---|------|---------|-------------|
| 109 | Vitamin A | C₂₀H₃₀O | Retinol, vision, cell differentiation |
| 110 | β-Carotene | C₄₀H₅₆ | Provitamin A, antioxidant pigment |
| 111 | Vitamin D₃ | C₂₇H₄₄O | Cholecalciferol, calcium homeostasis |
| 112 | Vitamin E | C₂₉H₅₀O₂ | α-Tocopherol, membrane antioxidant |
| 113 | Vitamin K₁ | C₃₁H₄₆O₂ | Phylloquinone, blood clotting |

### Random Generator

| # | Name | Description |
|---|------|-------------|
| 115 | Random | Procedurally generated molecule |

---

## Technical Details

- **Resolution**: 1024 × 768 pixels
- **Max atoms per molecule**: 200
- **Max bonds per molecule**: 250
- **Rendering**: CUDA ray tracing with Phong shading
- **Target GPU architecture**: sm_86 (RTX 30 series)

## Building

```bash
# From Visual Studio Developer Command Prompt
cd windows
nmake cuda_molecule
```

## Running

```bash
cuda_molecule.exe
```

---

*Part of CUDA-Demos-for-Jetson-Nano - A collection of CUDA graphics demonstrations*
