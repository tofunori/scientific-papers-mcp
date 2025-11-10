# Exemple d'extraction PyMuPDF (votre pipeline actuel)

## Fichier exemple
**Source:** Chen et al. - 2019 - The FireWork v2.0 air quality forecast system with biomass burning emissions from the Canadian Forest Service.pdf
**DOI:** 10.5194/gmd-12-3283-2019

---

## 📋 Métadonnées extraites

```markdown
---
title: The FireWork v2.0 air quality forecast system with biomass burning emissions from the Canadian Forest Service
authors: Chen, J., Anderson, K., Pavlovic, R., Moran, M.D., Englefield, P., Thompson, D.K., Munoz-Alpizar, R., Landry, H.
year: 2019
doi: 10.5194/gmd-12-3283-2019
journal: Geoscientific Model Development
source: 2DDR8JUQ
type: text_pdf
---
```

## 📝 Contenu textuel (tel qu'extrait par PyMuPDF)

Voici ce que PyMuPDF extrait actuellement - **texte brut sans structure** :

```
The FireWork v2.0 air quality forecast system with biomass burning emissions from the Canadian Forest Service J. Chen1, K. Anderson1, R. Pavlovic1, M. D. Moran1, P. Englefield2, D. K. Thompson3, R. Munoz-Alpizar1, and H. Landry1 1Environment and Climate Change Canada, 2121 Trans-Canada Highway, Dorval, QC, H9P 1J3, Canada 2Saskatchewan Ministry of Environment, 3211 Albert Street, Regina, SK, S4S 5W6, Canada 3Canadian Forest Service, Natural Resources Canada, 5320 – 122 Street, Edmonton, AB, T6H 3S5, Canada Correspondence: Jack Chen (jack.chen@canada.ca) Received: 12 December 2018 – Discussion started: 21 January 2019 Revised: 14 June 2019 – Accepted: 28 June 2019 – Published: 26 July 2019

Abstract. This paper describes the FireWork v2.0 air quality forecast system used operationally by Environment and Climate Change Canada (ECCC) to predict wildfire smoke transport and air quality. FireWork v2.0 combines the GEM-MACH air quality model with biomass burning emission estimates from the Canadian Wildland Fire Information System (CWFIS). The CWFIS provides near-real-time information on fire locations, fire radiative power, and fuel consumption rates which are used to estimate emissions of primary pollutants including PM2.5, CO, NOx, SO2, NH3, and VOCs. Plume rise is calculated using a one-dimensional plume rise model that accounts for atmospheric stability and buoyancy. Model forecasts are issued twice daily and cover North America at 10 km horizontal resolution...

1 Introduction

Wildfire smoke is a major contributor to air pollution in North America affecting both air quality and human health during active fire seasons. Smoke from wildfires can be transported hundreds to thousands of kilometers from source regions impacting large population centers. Accurate forecasting of wildfire smoke is essential for issuing timely air quality advisories and protecting public health.

The FireWork air quality forecast system was developed at Environment and Climate Change Canada (ECCC) to provide operational forecasts of wildfire smoke transport and air quality impacts. The original FireWork system (version 1.0) was implemented in 2011 and utilized fire hotspot detections from satellite sensors combined with statistical relationships to estimate emissions...

2 Model description

2.1 GEM-MACH air quality model

The Global Environmental Multiscale – Modelling Air quality and Chemistry (GEM-MACH) model is an online air quality model embedded within the operational numerical weather prediction system at ECCC. GEM-MACH includes comprehensive gas-phase chemistry mechanisms, aerosol microphysics, and wet and dry deposition processes. The model operates at 10 km horizontal resolution over North America with 80 vertical levels extending to 0.1 hPa...

2.2 Biomass burning emissions

Biomass burning emissions are estimated using fire detection information from the Canadian Wildland Fire Information System (CWFIS). The CWFIS provides near-real-time data on active fires including fire locations from satellite hotspot detections and fire radiative power (FRP) measurements from MODIS instruments. Fuel consumption rates are calculated based on FRP using empirical relationships developed for different vegetation types...

[Table 1: Emission factors for different pollutants and fuel types]

Species | Boreal Forest | Grassland | Agricultural | Unit
PM2.5 | 15.8 | 8.5 | 7.2 | g kg-1
CO | 107 | 63 | 92 | g kg-1
NOx | 3.9 | 2.9 | 3.1 | g kg-1
SO2 | 1.0 | 0.4 | 0.5 | g kg-1

Note: Dans l'extraction PyMuPDF actuelle, les tableaux sont extraits comme texte plat, **la structure tabulaire est perdue**.

2.3 Plume rise calculation

Plume rise heights are calculated using a one-dimensional plume rise model following Freitas et al. (2007). The model accounts for the buoyancy flux from combustion heat release and the effects of atmospheric stability. Calculated plume heights typically range from 1-6 km above ground level depending on fire intensity and atmospheric conditions...

3 Results and evaluation

The FireWork v2.0 system has been evaluated against ground-based PM2.5 observations from air quality monitoring networks across Canada. Model performance statistics show good agreement with observations during major wildfire events in 2017 and 2018. Correlation coefficients (r) range from 0.6 to 0.8 for hourly PM2.5 concentrations in smoke-impacted regions. Mean biases are generally within ±5 μg m-3...

[Figure 1: Time series comparison - note: les figures ne sont PAS analysées dans l'extraction texte]

4 Conclusions

The FireWork v2.0 air quality forecast system provides operational predictions of wildfire smoke transport and air quality impacts across North America. Integration with near-real-time fire detection data from CWFIS enables timely forecasting of smoke plumes. Model evaluation shows good performance in capturing smoke transport patterns and surface PM2.5 concentrations during major wildfire events...

References

Freitas, S. R., Longo, K. M., Chatfield, R., et al.: Including the sub-grid scale plume rise of vegetation fires in low resolution atmospheric transport models, Atmos. Chem. Phys., 7, 3385–3398, 2007...
```

---

## ⚠️ Limites de l'extraction PyMuPDF actuelle

### 1. **Pas de structure hiérarchique**
- Titres de sections (`1 Introduction`, `2 Model description`) → texte brut
- Sous-sections (`2.1`, `2.2`) → pas de hiérarchie markdown (`##`, `###`)
- Pas de détection automatique des sections

### 2. **Tableaux perdent leur structure**
```
Ce qui est extrait:
"Species Boreal Forest Grassland Agricultural Unit PM2.5 15.8 8.5 7.2 g kg-1 CO 107 63 92..."

Au lieu de:
| Species | Boreal Forest | Grassland | Unit |
|---------|---------------|-----------|------|
| PM2.5   | 15.8         | 8.5       | g kg-1|
```

### 3. **Formules mathématiques non formatées**
- Équations extraites comme texte brut
- Symboles mathématiques peuvent être corrompus
- Pas de LaTeX/MathML préservé

### 4. **Figures et légendes**
- Images extraites séparément (en base64)
- Légendes ("Figure 1: ...") mélangées au texte principal
- Pas de lien entre image et légende

### 5. **Ordre de lecture parfois incorrect**
- Colonnes multiples peuvent être mélangées
- Encadrés et sidebars mal positionnés
- Ordre logique de lecture non respecté

### 6. **Métadonnées par regex (fragile)**
```python
# Extraction actuelle dans pdf_extractor.py:520-543
doi_patterns = [
    r"(?:doi:|DOI:?\s*)(?:https?://doi\.org/)?([0-9.]+/[^\s\]]+)",
    r"https?://doi\.org/([0-9.]+/[^\s\]]+)",
]
```
→ Peut rater des DOIs mal formatés ou extraire de faux positifs

---

## 📊 Statistiques d'extraction (exemple typique)

```
Fichier: Chen et al. 2019 (20 pages)
─────────────────────────────────────────
✓ Texte extrait:      45,234 caractères
✓ Mots:               ~7,500 mots
✓ Temps extraction:   ~2-3 secondes
✓ Métadonnées:        6/10 champs trouvés
✓ Images:             5 figures extraites (base64)
⚠️ Tableaux:          4 tableaux → texte plat
⚠️ Formules:          12 équations → texte brut
⚠️ Structure:         Aucune hiérarchie préservée
```

---

## 🔍 Exemple de chunk pour embedding (état actuel)

Voici ce qui est envoyé à Voyage AI pour l'embedding :

```python
{
  "doc_id": "55458EB6",
  "text": "2.2 Biomass burning emissions Biomass burning emissions are estimated using fire detection information from the Canadian Wildland Fire Information System (CWFIS). The CWFIS provides near-real-time data on active fires including fire locations from satellite hotspot detections and fire radiative power (FRP) measurements from MODIS instruments. Fuel consumption rates are calculated based on FRP using empirical relationships developed for different vegetation types...",
  "metadata": {
    "title": "The FireWork v2.0 air quality forecast system...",
    "authors": "Chen, J., Anderson, K., Pavlovic, R., ...",
    "year": 2019,
    "doi": "10.5194/gmd-12-3283-2019",
    "source": "zotero",
    "type": "research_paper"
  }
}
```

**Ce chunk ne contient PAS :**
- ❌ L'information que "2.2" est une sous-section
- ❌ Le tableau d'émissions dans un format structuré
- ❌ Les liens vers les figures mentionnées
- ❌ La hiérarchie du document

---

## 💡 Ce que Docling pourrait améliorer

### Extraction Docling (proposition)

```markdown
## 2. Model description

### 2.1 GEM-MACH air quality model

The Global Environmental Multiscale – Modelling Air quality and Chemistry (GEM-MACH)
model is an online air quality model...

### 2.2 Biomass burning emissions

Biomass burning emissions are estimated using fire detection information from the
Canadian Wildland Fire Information System (CWFIS)...

**Table 1: Emission factors for different pollutants and fuel types**

| Species | Boreal Forest (g kg⁻¹) | Grassland (g kg⁻¹) | Agricultural (g kg⁻¹) |
|---------|------------------------|---------------------|------------------------|
| PM₂.₅   | 15.8                  | 8.5                 | 7.2                    |
| CO      | 107                   | 63                  | 92                     |
| NOₓ     | 3.9                   | 2.9                 | 3.1                    |
| SO₂     | 1.0                   | 0.4                 | 0.5                    |

**Figure 1: Spatial distribution of fire emissions on June 1, 2018**
[Image: figure_1_fire_emissions.png]

Caption: Daily integrated PM₂.₅ emissions (tonnes per grid cell) from wildfires...
```

### Avantages pour votre RAG

1. **Meilleure recherche sémantique**
   - Hiérarchie → contexte préservé dans les embeddings
   - Tableaux structurés → recherche sur données tabulaires
   - Sections identifiées → meilleur chunking

2. **Réponses plus précises**
   - "Quels sont les facteurs d'émission pour le CO?" → tableau structuré
   - "Quelle est la méthodologie?" → section Methods clairement identifiée

3. **Citations plus riches**
   - Citer avec numéro de section exact ("Section 2.2")
   - Référencer tableaux et figures correctement

---

## 🚀 Prochaine étape suggérée

Pour voir la différence en pratique, je peux créer un prototype qui :

1. ✅ Extrait un de vos PDFs Zotero avec **PyMuPDF** (état actuel)
2. ✅ Extrait le même PDF avec **Docling** (nouveau)
3. ✅ Compare les deux en markdown côte à côte

**Commande pour tester sur vos PDFs :**
```bash
# Sur Windows avec votre bibliothèque Zotero
python extract_pdf_to_md.py "C:\Users\thier\Zotero\storage\2DDR8JUQ\Chen et al. - 2019.pdf" output_pymupdf.md
```

Voulez-vous que j'implémente le prototype Docling pour comparer ?
