# ML
## Wheat Kernel Dataset

The dataset comprises kernels belonging to three different varieties of wheat: Kama, Rosa, and Canadian. The examined group consists of 70 elements for each variety, randomly selected for the experiment. The dataset is derived from high-quality visualization of the internal kernel structure using a soft X-ray technique. This technique is non-destructive and more cost-effective compared to other sophisticated imaging methods such as scanning microscopy or laser technology. The images were recorded on 13x18 cm X-ray KODAK plates. The studies were conducted using combine-harvested wheat grain from experimental fields explored at the Institute of Agrophysics of the Polish Academy of Sciences in Lublin.

### Purpose of the Dataset

The dataset is suitable for tasks related to classification and cluster analysis. The target attribute for classification is the **Seed Type**.

### Attribute Information

To construct the data, seven geometric parameters of wheat kernels were measured, all of which are real-valued continuous:

#### Numeric Attributes:

1. **Area (A)**
2. **Perimeter (P)**
3. **Compactness (C)**: Calculated as \( \frac{4 \pi A}{P^2} \)
4. **Length of Kernel**
5. **Width of Kernel**
6. **Asymmetry Coefficient**
7. **Length of Kernel Groove**

#### Discrete Attribute:

- **Seed Type**: The variety of wheat kernel (Kama, Rosa, Canadian).
