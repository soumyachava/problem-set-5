library(readr)
library(ggplot2)       
library(FactoMineR)    
library(factoextra)   
library(caret)
library(Rtsne)
library(cluster)
library(clValid)
library(clusterCrit)
library(dendextend) 

## 1. Load Data
data <- read.csv("Breast_Cancer.csv")
head(data)



# 2. PCA
# Converting diagnosis to factor
data$diagnosis <- as.factor(data$diagnosis)
# Removing categorical column for PCA
data_numeric <- data[, -1]
# Standardizing the numeric features
data_scaled <- scale(data_numeric)
# PCA
pca_result <- prcomp(data_scaled, center = TRUE, scale. = TRUE)
# Printing PCA summary
summary(pca_result)

# scree plot
fviz_eig(pca_result, addlabels = TRUE, barfill = "steelblue", barcolor = "black")

# Biplot
fviz_pca_biplot(pca_result, 
                label = "var", 
                habillage = data$diagnosis,
                addEllipses = TRUE, 
                palette = c("red", "blue"),
                repel = TRUE)

# variable contribution
fviz_pca_var(pca_result, col.var = "contrib", gradient.cols = c("blue", "green", "red"), repel = TRUE)

# Printing variable contributions
pca_var_contrib <- data.frame(pca_result$rotation[, 1:2])
# Contributions to PC1
pca_PC1_contrib <- pca_var_contrib[order(abs(pca_var_contrib$PC1), decreasing = TRUE), ]
print("Top contributors to PC1:")
print(pca_PC1_contrib)
# Contributions to PC2
pca_PC2_contrib <- pca_var_contrib[order(abs(pca_var_contrib$PC2), decreasing = TRUE), ]
print("Top contributors to PC2:")
print(pca_PC2_contrib)


#       Contributors to PC1:        PC1          PC2
       #concave.points_mean     -0.26085376  0.034767500    
       #concavity_mean          -0.25840048 -0.060165363
       #concave.points_worst    -0.25088597  0.008257235
       #compactness_mean        -0.23928535 -0.151891610
       #perimeter_worst         -0.23663968  0.199878428
       #concavity_worst         -0.22876753 -0.097964114
       #radius_worst            -0.22799663  0.219866379
       #perimeter_mean          -0.22753729  0.215181361
       #area_worst              -0.22487053  0.219351858
       #area_mean               -0.22099499  0.231076711
       #radius_mean             -0.21890244  0.233857132
       #perimeter_se            -0.21132592  0.089457234
       #compactness_worst       -0.21009588 -0.143593173
       #radius_se               -0.20597878  0.105552152
       #area_se                 -0.20286964  0.152292628
       #concave.points_se       -0.18341740 -0.130321560
       #compactness_se          -0.17039345 -0.232715896
       #concavity_se            -0.15358979 -0.197207283
       #smoothness_mean         -0.14258969 -0.186113023
       #symmetry_mean           -0.13816696 -0.190348770
       #fractal_dimension_worst -0.13178394 -0.275339469
       #smoothness_worst        -0.12795256 -0.172304352
       #symmetry_worst          -0.12290456 -0.141883349
       #texture_worst           -0.10446933  0.045467298
       #texture_mean            -0.10372458  0.059706088
       #fractal_dimension_se    -0.10256832 -0.280092027
       #fractal_dimension_mean  -0.06436335 -0.366575471
       #symmetry_se             -0.04249842 -0.183848000
       #texture_se              -0.01742803 -0.089979682
       #smoothness_se           -0.01453145 -0.204430453

#      Contributors to PC2:         PC1          PC2
       #fractal_dimension_mean  -0.06436335 -0.366575471
       #fractal_dimension_se    -0.10256832 -0.280092027
       #fractal_dimension_worst -0.13178394 -0.275339469
       #radius_mean             -0.21890244  0.233857132
       #compactness_se          -0.17039345 -0.232715896
       #area_mean               -0.22099499  0.231076711
       #radius_worst            -0.22799663  0.219866379
       #area_worst              -0.22487053  0.219351858
       #perimeter_mean          -0.22753729  0.215181361
       #smoothness_se           -0.01453145 -0.204430453
       #perimeter_worst         -0.23663968  0.199878428
       #concavity_se            -0.15358979 -0.197207283
       #symmetry_mean           -0.13816696 -0.190348770
       #smoothness_mean         -0.14258969 -0.186113023
       #symmetry_se             -0.04249842 -0.183848000
       #smoothness_worst        -0.12795256 -0.172304352
       #area_se                 -0.20286964  0.152292628
       #compactness_mean        -0.23928535 -0.151891610
       #compactness_worst       -0.21009588 -0.143593173
       #symmetry_worst          -0.12290456 -0.141883349
       #concave.points_se       -0.18341740 -0.130321560
       #radius_se               -0.20597878  0.105552152
       #concavity_worst         -0.22876753 -0.097964114
       #texture_se              -0.01742803 -0.089979682
       #perimeter_se            -0.21132592  0.089457234
       #concavity_mean          -0.25840048 -0.060165363
       #texture_mean            -0.10372458  0.059706088
       #texture_worst           -0.10446933  0.045467298
       #concave.points_mean     -0.26085376  0.034767500
       #concave.points_worst    -0.25088597  0.008257235


# confusion matrix
set.seed(123)
# Performing k-means clustering using the first two principal components
kmeans_pca <- kmeans(pca_result$x[, 1:2], centers = 2, nstart = 25)
# Converting actual labels to numeric
true_labels <- ifelse(data$diagnosis == "M", 1, 0)
# Getting the cluster labels
cluster_labels <- kmeans_pca$cluster
# Determining which cluster corresponds to Malignant (M) and which to Benign (B)
cluster_mapping <- ifelse(mean(true_labels[cluster_labels == 1]) > 0.5, 1, 0)
# Mapping clusters to actual labels
predicted_labels <- ifelse(cluster_labels == 1, cluster_mapping, 1 - cluster_mapping)
# Creating and printing the confusion matrix
conf_matrix_pca <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels))
print(conf_matrix_pca)
# Answer: Variance Explained: 
  # PC1 explains 44.27% of the variance, while PC2 explains 18.97%.
  # Together, the first two components explain 63.24% of the total variance.
  # Variables contributing the most to PC1 and PC2
# PC1: The most contributing variables (highest absolute values) are:
  #  concave.points_mean (-0.2609)
  #  concavity_mean (-0.2584)
  #  concave.points_worst (-0.2509)
  #  compactness_mean (-0.2393)
  #  perimeter_worst (-0.2366)
# PC2: The most contributing variables (highest absolute values) are:
  #  fractal_dimension_mean (-0.3666)
  #  fractal_dimension_se (-0.2801)
  #  fractal_dimension_worst (-0.2753)
  #  compactness_se (-0.2327)
  #  area_mean (0.2311)
# Confusion Matrix Interpretation
   #               Reference
   #     Prediction   0   1
   #              0 341  37
   #              1  16 175
#Accuracy : 0.9069          
#95% CI : (0.8799, 0.9294)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : < 2e-16         
#Kappa : 0.7967          
#Mcnemar's Test P-Value : 0.00601         
            #Sensitivity : 0.9552          
            #Specificity : 0.8255          
            #Pos Pred Value : 0.9021          
            #Neg Pred Value : 0.9162          
            #Prevalence : 0.6274          
            #Detection Rate : 0.5993          
            #Detection Prevalence : 0.6643          
            #Balanced Accuracy : 0.8903
# Accuracy: 90.69% (Good classification performance)
# Sensitivity (True Positive Rate for class 0): 95.52%
# Specificity (True Negative Rate for class 1): 82.55%
# The model predicts benign cases well but has some misclassifications for malignant cases.



# 3. tsne
# Running t-SNE with different perplexities and visualizing the results
perplexities <- c(10, 30, 50)
for (perplexity in perplexities) {
  tsne_result <- Rtsne(data_scaled, dims = 2, perplexity = perplexity, check_duplicates = FALSE)
  
  # Creating a data frame for visualization
  tsne_data <- data.frame(
    Dim1 = tsne_result$Y[, 1],
    Dim2 = tsne_result$Y[, 2],
    Diagnosis = factor(data$diagnosis)
  )
  
  # Plotting t-SNE result
  ggplot(tsne_data, aes(x = Dim1, y = Dim2, color = Diagnosis)) +
    geom_point(alpha = 0.7) +
    theme_minimal() +
    ggtitle(paste("t-SNE (Perplexity =", perplexity, ")"))
}

# Performing K-means clustering on the t-SNE results
set.seed(123)
kmeans_tsne <- kmeans(tsne_result$Y, centers = 2, nstart = 25)

# Maping the clusters to actual labels
cluster_labels <- kmeans_tsne$cluster
cluster_mapping <- ifelse(mean(true_labels[cluster_labels == 1]) > 0.5, 1, 0)
predicted_labels <- ifelse(cluster_labels == 1, cluster_mapping, 1 - cluster_mapping)

# Creating and printing the confusion matrix
conf_matrix_tsne <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels))
print(conf_matrix_tsne)
# Confusion matrix:        Reference
#              Prediction   0   1
#                        0 328  23
#                        1  29 189
#Accuracy : 0.9086         
#95% CI : (0.8819, 0.931)
#No Information Rate : 0.6274         
#P-Value [Acc > NIR] : <2e-16         
#Kappa : 0.8056         
#Mcnemar's Test P-Value : 0.4881         
           #Sensitivity : 0.9188         
           #Specificity : 0.8915         
           #Pos Pred Value : 0.9345         
           #Neg Pred Value : 0.8670         
           #Prevalence : 0.6274         
           #Detection Rate : 0.5764         
           #Detection Prevalence : 0.6169         
           #Balanced Accuracy : 0.9051
# The model demonstrates strong predictive ability with high accuracy, sensitivity, and specificity.
# The high kappa statistic confirms that predictions align well with actual labels.
# No significant bias in misclassification is observed, as indicated by McNemar’s test.



# 4. k-means clustering
# performing Elbow method
wcss <- vector()
for (i in 1:10) {
  kmeans_model <- kmeans(data_scaled, centers = i, nstart = 25)
  wcss[i] <- kmeans_model$tot.withinss
}
# Plotting the Elbow method
ggplot(data.frame(Clusters = 1:10, WCSS = wcss), aes(x = Clusters, y = WCSS)) +
  geom_line() +
  geom_point() +
  ggtitle("Elbow Method for Optimal Clusters") +
  xlab("Number of Clusters") +
  ylab("WCSS")

# Silhouette Score
# Perform K-means clustering
kmeans_result <- kmeans(data_scaled, centers = 2, nstart = 25)

# Getting mean Silhouette score
silhouette_score <- silhouette(kmeans_result$cluster, dist(data_scaled))
mean(silhouette_score[, 3]) 
# silhouette score: 0.344974

# Plotting silhouette score
fviz_silhouette(silhouette_score)

# Getting cluster labels
cluster_labels <- kmeans_result$cluster
# Mapping the clusters to the true labels
cluster_mapping <- ifelse(mean(true_labels[cluster_labels == 1]) > 0.5, 1, 0)
predicted_labels <- ifelse(cluster_labels == 1, cluster_mapping, 1 - cluster_mapping)
# Confusion Matrix
conf_matrix_kmeans <- confusionMatrix(as.factor(predicted_labels), as.factor(true_labels))
print(conf_matrix_kmeans)
# Answer: 
 # silhouette score: 0.344974
 #     cluster size ave.sil.width
        # 1  189          0.15
        # 2  380          0.44
# Accuracy of 91.04% suggests the model correctly classifies most cases.
# A sensitivity of 96.08% means the model correctly identifies most benign cases.
# A specificity of 82.55% indicates some misclassification of malignant cases.
# A p-value of 0.002066 suggests there is some bias in classification


# 5. Hierarchical Clustering
# Computing distance matrix
dist_matrix <- dist(data_scaled, method = "euclidean")  

# Performing hierarchical clustering with different linkage methods
hc_single <- hclust(dist_matrix, method = "single")    
hc_complete <- hclust(dist_matrix, method = "complete")  
hc_average <- hclust(dist_matrix, method = "average")  
hc_ward <- hclust(dist_matrix, method = "ward.D2")  

# Plotting dendrograms
par(mfrow = c(1,1))  
plot(hc_single, main = "Single Linkage", cex = 0.4)
plot(hc_complete, main = "Complete Linkage", cex = 0.4)  
plot(hc_average, main = "Average Linkage", cex = 0.4)  
plot(hc_ward, main = "Ward Linkage", cex = 0.4)  

# Cut tree to form 2 clusters
clusters_single <- cutree(hc_single, k = 2)  
clusters_complete <- cutree(hc_complete, k = 2)  
clusters_average <- cutree(hc_average, k = 2)  
clusters_ward <- cutree(hc_ward, k = 2)  

# Function to map clusters to actual labels
map_clusters <- function(predicted_clusters, true_labels) {
  cluster_mapping <- ifelse(mean(true_labels[predicted_clusters == 1]) > 0.5, 1, 0)
  return(ifelse(predicted_clusters == 1, cluster_mapping, 1 - cluster_mapping))
}

# Mapping clusters to actual labels
pred_single <- map_clusters(clusters_single, true_labels)
pred_complete <- map_clusters(clusters_complete, true_labels)
pred_average <- map_clusters(clusters_average, true_labels)
pred_ward <- map_clusters(clusters_ward, true_labels)

# Computing confusion matrices for each linkage
cm_single <- confusionMatrix(as.factor(pred_single), as.factor(true_labels))
cm_complete <- confusionMatrix(as.factor(pred_complete), as.factor(true_labels))
cm_average <- confusionMatrix(as.factor(pred_average), as.factor(true_labels))
cm_ward <- confusionMatrix(as.factor(pred_ward), as.factor(true_labels))

# Printing confusion matrices
print("Confusion Matrix - Single Linkage")
print(cm_single)

print("Confusion Matrix - Complete Linkage")
print(cm_complete)

print("Confusion Matrix - Average Linkage")
print(cm_average)

print("Confusion Matrix - Ward Linkage")
print(cm_ward)

# Single Linkage Confusion Matrix:
  #           Reference
  #Prediction   0   1
#           0 357 210
#           1   0   2
#Accuracy : 0.6309          
#95% CI : (0.5898, 0.6707)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : 0.4497          
#Kappa : 0.0118          
#Mcnemar's Test P-Value : <2e-16          
          #Sensitivity : 1.000000        
          #Specificity : 0.009434        
          #Pos Pred Value : 0.629630        
          #Neg Pred Value : 1.000000        
          #Prevalence : 0.627417        
          #Detection Rate : 0.627417        
          #Detection Prevalence : 0.996485        
          #Balanced Accuracy : 0.504717

# Complete Linkage Confusion Matrix:
  #                   Reference
  #         Prediction   0   1
#                    0 357 210
#                    1   0   2
#Accuracy : 0.6309          
#95% CI : (0.5898, 0.6707)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : 0.4497          
#Kappa : 0.0118          
#Mcnemar's Test P-Value : <2e-16          
          #Sensitivity : 1.000000        
          #Specificity : 0.009434        
          #Pos Pred Value : 0.629630        
          #Neg Pred Value : 1.000000        
          #Prevalence : 0.627417        
          #Detection Rate : 0.627417        
          #Detection Prevalence : 0.996485        
          #Balanced Accuracy : 0.504717     

# Average Linkage Confusion Matrix:
  #            Reference
  # Prediction   0   1
#             0 357 209
#             1   0   3
#Accuracy : 0.6327          
#95% CI : (0.5916, 0.6724)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : 0.4156         
#Kappa : 0.0177          
#Mcnemar's Test P-Value : <2e-16          
          #Sensitivity : 1.00000         
          #Specificity : 0.01415         
          #Pos Pred Value : 0.63074         
          #Neg Pred Value : 1.00000         
          #Prevalence : 0.62742         
          #Detection Rate : 0.62742         
          #Detection Prevalence : 0.99473         
          #Balanced Accuracy : 0.50708 

# Wards Linkage Confusion Matrix:
  #               Reference
  #    Prediction   0   1
#               0  337  48
#               1  20 164
#Accuracy : 0.8805        
#95% CI : (0.851, 0.906)
#No Information Rate : 0.6274        
#P-Value [Acc > NIR] : < 2.2e-16     
#Kappa : 0.7373        
#Mcnemar's Test P-Value : 0.001059      
          #Sensitivity : 0.9440        
          #Specificity : 0.7736        
          #Pos Pred Value : 0.8753        
          #Neg Pred Value : 0.8913        
          #Prevalence : 0.6274        
          #Detection Rate : 0.5923        
          #Detection Prevalence : 0.6766        
          #Balanced Accuracy : 0.8588        

# Confusion Matrix Evaluation: 
# Single, Complete, and Average Linkage methods have similar results: low accuracy (~63%), high sensitivity for class 0 (100%), but poor specificity for class 1 (low detection of class 1).
# Ward Linkage performs much better with 88% accuracy, higher sensitivity (94.4%) and specificity (77.4%), and a strong Kappa value (0.7373), indicating better overall clustering performance.
# Conclusion: Ward Linkage outperforms the others in terms of accuracy and separation between classes.



# 6. combining methods
# Using first 10 principal components for t-SNE input
pca_tsne_input <- pca_result$x[, 1:10]
set.seed(123)  
tsne_result_combined <- Rtsne(pca_tsne_input, dims = 2, perplexity = 30, check_duplicates = FALSE)

# Creating a data frame for visualization
tsne_data_combined <- data.frame(
  Dim1 = tsne_result_combined$Y[, 1],
  Dim2 = tsne_result_combined$Y[, 2],
  Diagnosis = factor(data$diagnosis)
)
set.seed(123)
kmeans_combined <- kmeans(tsne_result_combined$Y, centers = 2, nstart = 25)

# K-means cluster labels
cluster_labels_combined <- kmeans_combined$cluster
cluster_mapping_combined <- ifelse(mean(true_labels[cluster_labels_combined == 1]) > 0.5, 1, 0)
predicted_labels_combined <- ifelse(cluster_labels_combined == 1, cluster_mapping_combined, 1 - cluster_mapping_combined)
# confusion matrix for pca + tsne + kmeans
conf_matrix_combined <- confusionMatrix(as.factor(predicted_labels_combined), as.factor(true_labels))
print(conf_matrix_combined)
# confusion matrix:   Reference
          # Prediction   0   1
          #          0 322  20
          #          1  35 192
#Accuracy : 0.9033         
#95% CI : (0.876, 0.9264)
#No Information Rate : 0.6274         
#P-Value [Acc > NIR] : < 2e-16        
#Kappa : 0.7962         
#Mcnemar's Test P-Value : 0.05906        
           #Sensitivity : 0.9020         
           #Specificity : 0.9057         
           #Pos Pred Value : 0.9415         
           #Neg Pred Value : 0.8458         
           #Prevalence : 0.6274         
           #Detection Rate : 0.5659         
           #Detection Prevalence : 0.6011         
           #Balanced Accuracy : 0.9038         
           #'Positive' Class : 0      

# combining PCA + Hierarchical Clustering
dist_pca <- dist(pca_tsne_input)
hc_pca <- hclust(dist_pca, method = "ward.D2")
clusters_pca_hc <- cutree(hc_pca, k = 2)

pred_pca_hc <- map_clusters(clusters_pca_hc, true_labels)
conf_matrix_pca_hc <- confusionMatrix(as.factor(pred_pca_hc), as.factor(true_labels))
print("PCA + Hierarchical Clustering:")
print(conf_matrix_pca_hc)
# confusion matrix pca+ hierarchical Clustering
                    #Reference
#       Prediction   0   1
#                 0 318   7
#                 1  39 205
#Accuracy : 0.9192          
#95% CI : (0.8936, 0.9402)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : < 2.2e-16       
#Kappa : 0.8322          
#Mcnemar's Test P-Value : 4.861e-06       
           #Sensitivity : 0.8908          
           #Specificity : 0.9670          
           #Pos Pred Value : 0.9785          
           #Neg Pred Value : 0.8402          
           #Prevalence : 0.6274          
           #Detection Rate : 0.5589          
           #Detection Prevalence : 0.5712          
           #Balanced Accuracy : 0.9289       

# Combining t-SNE + Hierarchical Clustering
set.seed(123)
tsne_result_hc <- Rtsne(data_scaled, dims = 2, perplexity = 30, check_duplicates = FALSE)
dist_tsne <- dist(tsne_result_hc$Y)
hc_tsne <- hclust(dist_tsne, method = "ward.D2")
clusters_tsne_hc <- cutree(hc_tsne, k = 2)

pred_tsne_hc <- map_clusters(clusters_tsne_hc, true_labels)
conf_matrix_tsne_hc <- confusionMatrix(as.factor(pred_tsne_hc), as.factor(true_labels))
print("t-SNE + Hierarchical Clustering:")
print(conf_matrix_tsne_hc)
# confusion matrix tsne + hierarchical Clustering
                #Reference
    #Prediction   0   1
#              0 272   6
#              1  85 206
#Accuracy : 0.8401          
#95% CI : (0.8073, 0.8692)
#No Information Rate : 0.6274          
#P-Value [Acc > NIR] : < 2.2e-16       
#Kappa : 0.682           
#Mcnemar's Test P-Value : 2.919e-16       
          #Sensitivity : 0.7619          
          #Specificity : 0.9717          
          #Pos Pred Value : 0.9784          
          #Neg Pred Value : 0.7079          
          #Prevalence : 0.6274          
          #Detection Rate : 0.4780          
          #Detection Prevalence : 0.4886          
          #Balanced Accuracy : 0.8668 




# does Combinig Method work better?
# The individual methods like PCA, t-SNE, and k-means performed quite well overall. For example, PCA followed by k-means gave an accuracy of 90.33% and a Kappa of 0.7962, which indicates a good level of agreement with the actual labels.
# When we combined methods, the results varied. The combination of PCA and hierarchical clustering stood out, achieving the highest accuracy of 91.92% and a Kappa of 0.8322. It also had strong sensitivity (89.08%) and very high specificity (96.70%), meaning it did a great job at identifying both malignant and benign cases.
# On the other hand, the t-SNE plus hierarchical clustering combo didn’t perform as well. Its accuracy dropped to 84.01% and sensitivity to 76.19%, so it missed more malignant cases than the other methods.
# The original combined approach—PCA followed by t-SNE and then k-means—performed decently, with the same accuracy (90.33%) as PCA + k-means, and balanced accuracy of 90.38%. So, while it didn’t outperform everything, it still held up well.
# In conclusion, combining methods can definitely improve performance, but only if the right pairings are chosen. In this case, PCA with hierarchical clustering gave the best overall results across the board.



# 7. Evaluation
print(conf_matrix_kmeans)       
print(conf_matrix_pca)       
print(conf_matrix_tsne)      
print(conf_matrix_combined)  

# Dunn Index for K-means on raw data
dunn_raw <- dunn(dist(data_scaled), kmeans_result$cluster)
print(dunn_raw)
# Dunn Index for K-means on raw data: 0.06076299

# Dunn Index for PCA + K-means
dunn_pca <- dunn(dist(pca_result$x[, 1:2]), kmeans_pca$cluster)
print(dunn_pca)
# Dunn Index for PCA: 0.00933765

# Dunn Index for t-SNE + K-means
dunn_tsne <- dunn(dist(tsne_result$Y), kmeans_tsne$cluster)
print(dunn_tsne)
# Dunn Index for t-SNE: 0.03909667

# Dunn Index for PCA + t-SNE + K-means
dunn_combined <- dunn(dist(tsne_result_combined$Y), kmeans_combined$cluster)
print(dunn_combined) 
# Dunn Index combined: 0.01423854


# DB Index for K-means on raw data
db_raw <- intCriteria(as.matrix(data_scaled), kmeans_result$cluster, crit = "Davies_Bouldin")
print(db_raw)
# DB Index for K-means on raw data: 1.31232

# DB Index for PCA + K-means
db_pca <- intCriteria(as.matrix(pca_result$x[, 1:2]), kmeans_pca$cluster, crit = "Davies_Bouldin")
print(db_pca)
# DB Index for PCA: 0.8467404

# DB Index for t-SNE + K-means
db_tsne <- intCriteria(as.matrix(tsne_result$Y), kmeans_tsne$cluster, crit = "Davies_Bouldin")
print(db_tsne)
# DB Index for t-SNE: 0.6193636

# DB Index for PCA + t-SNE + K-means
db_combined <- intCriteria(as.matrix(tsne_result_combined$Y), kmeans_combined$cluster, crit = "Davies_Bouldin")
print(db_combined)
# combined DB Index: 0.6576984

#Confusion Matrices Comparison and Analysis:
 # The confusion matrices for K-means on raw data, PCA + K-means, t-SNE + K-means, and PCA + t-SNE + K-means are mostly similar in terms of accuracy (around 91%),sensitivity, and specificity
 # However, the combined approach (PCA + t-SNE + K-means) shows a slightly lower accuracy (90.33%).
 # The reason for this small difference is that when you combine PCA and t-SNE, you're adding more complexity. PCA reduces the data’s dimensions by keeping the main patterns, 
 # but t-SNE focuses on local patterns, which can sometimes interfere with the overall structure. So, when you add both, it could introduce a bit of distortion, 
 # which can lead to K-means misclassifying a few more points, bringing the accuracy down just a little bit.
# Dunn Index Comparison and Analysis:
 # K-means on raw data has the highest Dunn index (0.0608), which suggests the clusters are pretty well-separated.
 # PCA + K-means, however, has a much lower Dunn index (0.0093), PCA + t-SNE + K-means has the lowest Dunn index (0.0142).
 # The difference here is because of how the dimensionality reduction techniques like PCA and t-SNE affect the data. PCA keeps the global patterns but may lose some finer local details, 
 # while t-SNE focuses on those local patterns but messes with the global ones. So when you combine both, the data structure can get a bit distorted, 
 # which leads to worse separation between clusters, and that’s why the Dunn index goes down for the combined methods.
# Davies-Bouldin Index Comparison and Analysis:
 # K-means on raw data has the highest DB index (1.3123), PCA + t-SNE + K-means has a DB index of 0.6577, t-SNE + K-means has the lowest DB index (0.6194).
 # The difference in the DB index comes down to the effect of PCA and t-SNE. PCA helps by reducing dimensions and making the global structure clearer, which improves clustering. 
 # t-SNE is more focused on keeping the local structure intact, so it helps create tighter, well-separated clusters.
 # PCA can still improve things by showing the big picture, but t-SNE can introduce some distortions when you’re looking at the local structure. 



# 8. Conclusion of analysis:
# Based on the clustering analysis, the results reveal some interesting findings about the different methods used.
# Based on the clustering analysis, the results reveal some interesting findings about the different methods used.
# This shows that K-means can effectively classify the data.
# When dimensionality reduction techniques like PCA and t-SNE were applied before clustering, the accuracy slightly dropped to around 90%, but it still performed well, highlighting the value of reducing dimensions for clearer clustering. 
# Among hierarchical clustering methods, Ward’s linkage stood out, with an accuracy of 88% and a balanced performance across sensitivity and specificity. 
# This suggests it helps create more compact and well-separated clusters. 
# combining methods can definitely improve performance, but only if the right pairings are chosen. In this case, PCA with hierarchical clustering gave the best overall results across the board.
# However, the Dunn Index, which measures how distinct the clusters are, was generally low, indicating that the separation between the clusters wasn’t as strong as we’d hoped.
# The confusion matrix also showed that while most models did well identifying the majority class, some methods like hierarchical clustering with average linkage struggled a bit with specificity.
# Overall, K-means and tsne + K-means were the top performers.


# 9.Additional questions
# 1. principal components for 80% varience
# PCA result from previous code
pca_summary <- summary(pca_result)
# Cumulative variance explained
cumulative_variance <- cumsum(pca_summary$importance[2,])
# Find the number of components required to explain 80% of the variance
components_80_var <- which(cumulative_variance >= 0.80)[1]
print(components_80_var)
# Answer: Number of principle components required to explain 80% of the variance is "5".

# 2.t-SNE preserves global distances between samples.
#Answer: False
# t-SNE is a technique that focuses on keeping nearby points in your data close together. 
# It does a good job of showing how similar points are to each other. 
# However, it doesn't worry too much about how far apart distant points are, so it may not always show the true overall distance between faraway points in the data.

# 3.Why do we scale the variables in this dataset?
# Answer: Scaling is important because the features in the data might have different units or ranges. 
# For example, one feature might go from 0 to 1000, while another goes from 0 to 1. 
# If we don’t scale the data, the feature with the larger range could have more influence on the results, especially in techniques like PCA, K-means, and t-SNE, which are sensitive to how big the numbers are. 
# Scaling makes sure that all features are treated equally and helps the analysis give more meaningful results.


# 4. Which metric favors high separation and low intra-cluster spread?
# Answer: A. Dunn Index
