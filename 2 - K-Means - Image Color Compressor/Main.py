import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
from numpy import asarray

 
# Number of colors that the final image will have
number_of_colors = 24

image = Image.open('parrot.png')
image = image.convert('RGB')
data = asarray(image)

# Reshapping image matrix pixels so that they are ordered in a line
train = data.reshape(data.shape[0]*data.shape[1],3)


kmeans = KMeans(n_clusters = number_of_colors, random_state = 0, n_init='auto')
kmeans.fit(train)

centroids = np.round(kmeans.cluster_centers_)
result = centroids[kmeans.labels_]
result = result.reshape(data.shape[0],data.shape[1],3).astype(int)


# In[23]:


plt.imshow(result)
plt.show()
plt.imshow(data)
plt.show()


# In[24]:


print(kmeans.inertia_/train.shape[0])


# In[26]:


print(np.unique(centroids[kmeans.labels_],axis=0),centroids)

