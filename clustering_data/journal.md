### K = 3

|    |cluster size|     age |   health |   gender |   income level |   number of dependents |   survival with jacket |   survival delta |
|---:|-----------:|--------:|---------:|---------:|---------------:|-----------------------:|-----------------------:|-----------------:|
|  0 |         271| 7       |  5.57196 |  1.53506 |        7.29889 |                1.2952  |                7.80074 |          8.08118 |
|  1 |         156| 7.48077 |  4.8141  |  2.28846 |        5.88462 |                1.33974 |                3.53846 |          3.3141  |
|  2 |         246| 7.13415 |  6.7561  |  6.68699 |        7.53659 |                6.05285 |                6.78455 |          6.81301 |


### K = 4
|    |cluster size|     age |   health |   gender |   income level |   number of dependents |   survival with jacket |   survival delta |
|---:|-----------:|--------:|---------:|---------:|---------------:|-----------------------:|-----------------------:|-----------------:|
|  0 |         227| 7.09692 |  6.55947 |  6.77974 |        7.60793 |                6.38326 |                6.82379 |          6.78414 |
|  1 |         142| 7.1338  |  2.80986 |  1.65493 |        7.03521 |                1.09155 |                8.02817 |          8.26056 |
|  2 |         188| 7.00532 |  8.05319 |  1.94681 |        7.25    |                1.54787 |                6.95745 |          7.20213 |
|  3 |         116| 7.56897 |  4.49138 |  2.39655 |        5.69828 |                1.32759 |                2.91379 |          2.72414 |


## Note:
- There seems to be some features where the clusters actually vary, such as health, gender and number of dependents and survival with.
- The Silhouette method suggests that K = 3 lets us have the lower mean distance from each cluster compared to the other clusters, but the distribution seems to be pretty even for K = 4 as well.

## What can we do with these clusters?
- These clusters are generated from user given importance scores, so we will not be able to identify each user's cluster as they start the survey (these were gathered at the end of the survey)
- We can have a pool of questions that add more variance to the varying features (gender,health etc) to start off the survey so that we have a more varying input for our adaptive survey.
- The adaptive generation part could also reference this cluster to generate questions that may give us more information from a user. (i.e. a user whos profile fits cluster 1 in K = 4, we would ask a questions weighted between gender and number of dependents)

## Adding user features to the cluters

### K = 3 with user age

|    |   Agent:agegroup |     age |   health |   gender |   income level |   number of dependents |   survival with jacket |   survival delta |
|---:|-----------------:|--------:|---------:|---------:|---------------:|-----------------------:|-----------------------:|-----------------:|
|  0 |          45.4264 | 7.37984 |  5.84496 |  2.71318 |        6.69767 |                1.97674 |                6.30233 |          6.62791 |
|  1 |          30      | 7.01465 |  5.69963 |  3.59341 |        6.98168 |                2.90842 |                6.25275 |          6.20879 |
|  2 |          19.3506 | 7.20295 |  5.95203 |  4.01107 |        7.30627 |                3.69004 |                6.69742 |          6.76384 |


### K = 3 with user education

|    |   Agent:education |     age |   health |   gender |   income level |   number of dependents |   survival with jacket |   survival delta |
|---:|------------------:|--------:|---------:|---------:|---------------:|-----------------------:|-----------------------:|-----------------:|
|  0 |          0.924051 | 7.50633 |  4.85443 |  2.3038  |        5.87342 |                1.32278 |                3.5443  |          3.36709 |
|  1 |          1.28455  | 7.12602 |  6.7561  |  6.66667 |        7.54878 |                6.07317 |                6.78862 |          6.81301 |
|  2 |          0.988848 | 6.98885 |  5.5539  |  1.53903 |        7.30483 |                1.28625 |                7.82528 |          8.0855  |

### K = 3 with user gender

|    |   Agent:gender |     age |   health |   gender |   income level |   number of dependents |   survival with jacket |   survival delta |
|---:|---------------:|--------:|---------:|---------:|---------------:|-----------------------:|-----------------------:|-----------------:|
|  0 |       0.314286 | 7.12653 |  6.75102 |  6.68571 |        7.54694 |                6.07755 |                6.78776 |          6.81633 |
|  1 |       0.275093 | 6.98513 |  5.55762 |  1.54275 |        7.2974  |                1.30112 |                7.83271 |          8.08922 |
|  2 |       0.402516 | 7.50943 |  4.86792 |  2.2956  |        5.89937 |                1.32075 |                3.55346 |          3.37736 |

### notes:
- Sadly it seems like there is not a high correlation between the user features and their variance in features when I include the agent feature in the clustering.
- Another thing I can do is analyze the existing clusters that were generated without the user features and check to see if there is a correlation there (heatmaps)

### Non parametric active learning
- Need to find 'unlabled' feature value

### Algorithm idea:
- input: ```{'age':[-10:-5],'dependents':[-2:3], ...}```, number of alternatives N
- ouput: N alternatives that fit the input parameter as closely as possible without violating 'bad rules'

