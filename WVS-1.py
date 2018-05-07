import pandas as pd
import numpy as np
import sklearn.cluster
import scipy.stats
import matplotlib.pyplot as plt

def represents_int(s):
    '''
    represents_int determines whether its argument is a string that
    represents and integer.
    '''
    try: 
        int(s)
        return True
    except ValueError:
        return False

def read_spec(filename):
    '''
    read_spec reads the World Value Survey .sts specification file for "Wave 6."
    It returns a series where the index is a code for each column and 
    the value is a descriptive string.
    This file contains code descriptions for the 430 codes in Wave 6.
    '''
    s = pd.Series([])
    with open(filename, "r") as file:
        for line in file:
            # An example line from the spec file is:
            #    V4 5 (F2) [<= -1] {Important in life: Family} \V4
            # All we care about from this line is arr[0], which is "V4", 
            # and what's in brackets {}
            arr = line.split()
            if len(arr) < 2 or not represents_int(arr[1]):
                continue
            # The "code" is a string of letters and numbers, like "V4" or "MN_228R" 
            code = arr[0]
            # Need to parse the code description in between the { and }
            # There may be spaces inside the description, so the description may span
            # several elements
            desclist = [i for i, x in enumerate(arr) if "{" in x or "}" in x]
            # Some lines might not be a description of a code; they won't have brackets { }
            if not desclist:
                continue
            desc = ""
            min = desclist[0]
            max = desclist[-1]
            for i in range(min, max + 1):
                desc += arr[i]
                if i < max:
                    desc += " "
            # Append the code: desc pairs one by one
            sapp = pd.Series({code: desc})
            s = s.append(sapp)
    return s

def subcluster(X):
    '''
    This function is used to take a single cluster and break it into one or two subclusters.
    This is useful because our clusters will contain both correlated and anticorrelated
    components.  We want to distinguish between these sets of components.
    Note, however, that we cannot interpret this result without resorting to a manual
    check of the .sts specifications file to see whether that survey item
    had low numbers = most positive opinion or high numbers = most positive opinion
    '''
    kmeans = sklearn.cluster.KMeans(n_clusters=2)
    kmeans.fit(X)
    p = kmeans.predict(X)
    c = np.corrcoef([row for row in X])
    # Add the R-squared values for every cross-cluster pair
    # with a minus sign when R is negative.  If the result is negative, it means
    # the clusters are anticorrelated and there should be two clusters.  
    # Otherwise, there should be one.
    csum = sum(np.sign(c[i,j])*c[i, j]**2 
                for i, v in enumerate(p) 
                for j, w in enumerate(p) if v != w)   
    #The following code can be used to check the number of subclusters in each cluster
    #string = "Mini-cluster sum: " + str(csum)
    #if csum > 0:
    #    string += "(1 cluster)"
    #else:
    #    string += "(2 clusters)"
    #print(string)
    if csum < 0:
        return p # There should be two clusters
    else:
        return np.zeros(len(X)) # There should be only one cluster

def find_bad_nums(df, numrows, threshold, min, max):
    '''
    Given a DataFrame, find the numbers of the columns where more than
    10% of the rows are -4.  This means they were not included in the
    survey for 10% or more of the respondents.  I don't want to attempt to fill
    in the -4's with mean values if there are too many -4's.
    '''
    badlst = []
    for n in range(min, max):
        try:
            frac = df[df.columns[n]].value_counts()[-4] / numrows
            if frac > 0.1:
                badlst.append(n)
        except KeyError:
            pass
    return badlst

def custom_dist(row1, row2):
    '''
    The custom distance is the minimum of the usual distance
    along with the distance between one and the negative of the other.
    '''
    return min(
        np.sqrt(np.sum((row1 - row2)**2)),
        np.sqrt(np.sum((row1 + row2)**2))
    )

def get_all_nums(bad_nums):
  # These are the numbers for Inglehart's traditional vs. secular-rational items
  # I had to look these up manually from Inglehart's book
  trad_nums = [173, 344, 236, 392, 391, 10, 51, 170, 26, 
               112, 168, 240, 9, 239, 88, 99, 237, 49, 5, 150]
  trad_nums = np.array(trad_nums)-1 # column number equals printed number minus one
  trad_nums = [a for a in trad_nums if a not in bad_nums]

  # These are the numbers of Inglehart's survival vs. self-expression items        
  # I had to look these up manually from Inglehart's book
  surv_nums = [343, 11, 234, 89, 25, 53, 62, 56, 41, 39, 40, 101, 
               12, 47, 102, 57, 54, 43, 14, 16, 17, 7, 6, 148, 90, 161]
  surv_nums = np.array(surv_nums)-1
  surv_nums = [a for a in surv_nums if a not in bad_nums]

  # Group the two lists together.  Let's see if clustering can separate them again
  all_nums = trad_nums + surv_nums
  return (all_nums, trad_nums, surv_nums)

def get_zscore_matrix(df, numrows, all_nums):
  '''
  Given a DataFrame, a number of rows, and a list of column numbers (all_nums)
  Generate a zscore matrix, which is transposed relative to the DataFrame.
  The zscore matrix has rows that add to zero, their variance is 1, and each
  row in the zscore matrix corresponds to a column of the DataFrame.
  '''
  # Turn the df columns into rows (transpose) and select those rowse corresponding to all_nums
  X = df2.iloc[0:numrows].T.iloc[all_nums].as_matrix() # matrix of questions from start to end
  # insert nan wherever X < 0.  Note, this will trigger a runtime warning
  # saying that X < 0 results in some nan value where X is already nan. 
  # That's fine, those values are already nan and will remain nan.
  X[X < 0] = np.nan 
  # compute the mean of each survey item, averaging over all respondents
  row_means = np.nanmean(X, axis=1)
  # find the indices of the nans so that we can replace them by the mean value
  inds = np.where(np.isnan(X))
  # insert the row means in place of the nans
  X[inds] = np.take(row_means, inds[0])
  # Replace the respondents' answers by their z-scores for each survey item.
  # This is necessary so that we can perform a cluster analysis, since different
  # survey items are on wildly different scales.
  X = scipy.stats.zscore(X, axis=1)
  return X

def get_subc_list(X, labels):
  '''
  Get a list of subclusters for each cluster.
  Each item within a cluster is coded as either 0 or 1 using the subcluster function
  defined in this file.  Please see the subcluster function for further details.
  '''
  # indlist = [[row indices of X where cluster label = 0], [indices where label = 1], ...]
  indlist = [np.where(labels == a) for a in sorted(np.unique(labels))]
  # subXlist = [X rows where label = 0, X rows where label = 1, ... ]
  subXlist = [X[inds[0]] for inds in indlist] # inds will be a 1-elt. tuple, so take inds[0]
  # inv = [[predictions for label = 0], [predictions for label = 1], ...]
  # each prediction will be either 0 or 1 depending on how this item is inverted relative to
  # the other items in the cluster.  All the 0's are correlated with each other and
  # anticorrelated with the 1's.
  subc_list = [subcluster(subX) for subX in subXlist]
  return subc_list

def plot_pair_respondents(m, n, labels, matrix, spec_series, all_nums):
  '''
  Plot respondents m and n's z-scores
  '''
  clusterlist = [[a[0] for a in enumerate(labels) if a[1] == i] for i in range(0, 4)]

  plt.figure(figsize=(10,10))

  col = ['black', 'blue', 'red', 'orange']

  for i, c in enumerate(clusterlist):
    plt.plot(matrix[c,m], matrix[c,n], 'o', label = 'cluster ' + str(i), color = col[i])

  plt.title("Four clusters with respondents " + str(m) + " and " + str(n))
  plt.legend()
  plt.xlabel("z score of respondent " + str(m), size='large')
  plt.ylabel("z score of respondent " + str(n), size='large')

  #print(clusterlist)
  #print(spec_series[all_nums])

  for c in enumerate(clusterlist):
    for i in c[1]:
      plt.annotate(spec_series.iloc[all_nums[i]], (matrix[i,m] + 0.05, matrix[i,n]), color = col[c[0]])

  plt.savefig("respondents-" + str(m) + "-" + str(n) + ".jpg")

def plot_pair_items(m, n, mparity, nparity, matrix, spec_series, all_nums):
  '''
  Plot survey items m and n's z-scores
  There should almost always be points at z = 0; these correspond to missing data. 
  '''

  plt.figure(figsize=(10,8))

  matm = matrix[m, :] if mparity == 1 else -matrix[m, :]
  matn = matrix[n, :] if nparity == 1 else -matrix[n, :]

  nonzero = list(zip(*[a for a in zip(matm, matn) if a[0] != 0 and a[1] != 0]))
  mzer = zip(*[a for a in zip(matm, matn) if a[0] == 0 and a[1] != 0])
  nzer = zip(*[a for a in zip(matm, matn) if a[0] != 0 and a[1] == 0])
  zer = zip(*[a for a in zip(matm, matn) if a[0] == 0 and a[1] == 0])    

  alpha = 0.01
  z = 'green'
  nz = 'blue'
  plt.plot(*nonzero, color = nz, marker = 'o', alpha = alpha, linestyle='None')
  plt.plot(*mzer, color=z, marker='o', alpha = alpha, linestyle='None')
  plt.plot(*nzer, color=z, marker='o', alpha = alpha, linestyle='None')
  plt.plot(*zer, color=z, marker='o', alpha = alpha, linestyle='None')

  x, y = matm, matn
  plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'b-')

  plt.title("Correlation of: " + spec_series.iloc[all_nums[m]] + " and " + spec_series.iloc[all_nums[n]])
  plt.xlabel("z score of " + spec_series.iloc[all_nums[m]], size='large')
  plt.ylabel("z score of " + spec_series.iloc[all_nums[n]], size='large')
    
  plt.savefig("items-" + str(m) + "-" + str(n) + ".jpg")

print("Note: this program will write the results to output.txt")
print("Reading csv file ...")

# Read DataFrame df from csv file
# Note: some of these elements will be NaN because the csv file has some missing values,
# i.e. two adjacent commas with nothing between them.
df = pd.read_csv("F00005811-WV6_Data_ascii_delimited_v_2016_01_01/WV6_Data_ascii_delimited_v_2016_01_01.dat")

print("Reading sts file ...")

# Specification (.sts) file with description string for each survey item
specfile = "F00005811-WV6_Data_ascii_delimited_v_2016_01_01/WV6_Data_ascii_delimited_v_2016_01_01.sts"

spec_series = read_spec(specfile)

# Set column names to the codes from the .sts specifications file
df.columns = spec_series.index

while 1:
  print("Would you like to analyze:")
  print("(a) the world, two clusters")
  print("(b) the world, four clusters")
  print("(c) United States, two clusters")
  print("(d) United States, four clusters")
  letter = input(">")
  if letter in list("abcd"):
    break
  else:
    print("Input error: please enter a letter a-d")

dpref = {"a": -700, "b": -500, "c": -110, "d": -75}
dcode = {"c": 840, "d": 840}
dname = {"a": "the world", "b": "the world", "c": "the United States", "d": "the United States"}
numcols = len(df.columns)

if letter == "a" or letter == "b":
  df2 = df.copy() # the world
else:
  df2 = df[df['V2'] == dcode[letter]].copy() # U.S. only
numrows = len(df2.index)

# These are the numbers for "bad" survey items (columns) where more than 10% of the data is NaN.
# Many of these have much more than 10% being NaN.
# min survey item = 4 because the first four items in the list are not really survey items
bad_nums = find_bad_nums(df, numrows, threshold = 0.1, min = 4, max=numcols)

all_nums, trad_nums, surv_nums = get_all_nums(bad_nums)

all_nums.sort()

print("The number of good survey items for", dname[letter], "is: ", len(all_nums))
print("The number of respondents for", dname[letter], "is: ", numrows)

# The preference value for Affinity Propagation has to be set via trial and error.
# The smaller it is, the fewer clusters will appear.
# If it is too small, there will be only one cluster.
clust = sklearn.cluster.AffinityPropagation(affinity="precomputed", preference = dpref[letter])

X = get_zscore_matrix(df2, numrows, all_nums)

# Compute affinity matrix.  The affinity should be small when two items are far apart.
# That means we need to take the negative of the distance between the items.
# Also, we want to note correlations _and_ anticorrelations.  That means we should
# compute both the distance between row1 and row2 _and_ the distance between
# row1 and -row2.  We then take the minimum of these as the user-defined distance.
# In other words, [1, 0, -1, 0] should be "the same as" [-1, 0, 1, 0].
Xaff = -np.array(
                  [[custom_dist(row1, row2) for row1 in X] 
                  for row2 in X]
                )

clust.fit(Xaff)
labels = clust.labels_

# Now we have to determine if each survey item in each cluster was correlated or anticorrelated
# with the other survey items
subc_list = get_subc_list(X, labels)

msg = '''
How to interpret these results:

Clustering was performed by Affinity Propagation.
The number 0, 1, 2, or 3 is the cluster number.  All 0's are in the same cluster.
Surv-Exp means that this item is classified as Survival vs. Self-Expression by Inglehart.
Trad-Sec means that this item is classified as Traditional vs. Secular by Inglehart.
The phrase in brackets { } describes the nature of the item.

A's are correlated with other A's and anticorrelated with B's in that cluster.
However, to interpret this you need to know whether the scale on that item runs from e.g.
1 (highest) to 5 (lowest) or the reverse.  This requires manually consulting the tables located
in F00005811-WV6_Data_ascii_delimited_v_2016_01_01/WV6_Data_ascii_delimited_v_2016_01_01.sts.

'''

with open("output.txt", "w") as file:
  file.write(msg)
  print(msg)  

  predlist = [(a[1],
             "Trad-Sec" if all_nums[a[0]] in trad_nums else "Surv-Exp",
             spec_series.iloc[all_nums[a[0]]],
             all_nums[a[0]]
            ) for a in enumerate(labels)]
  # Sort by cluster, then by survey item number
  predlist.sort(key = lambda x: (x[0], x[3]))
  joinedlist  = []
  for i in subc_list:
      joinedlist = joinedlist + list(i)
  # joinedlist is sorted in the same way as predlist, so they can be zipped
  for x, i in zip(predlist, joinedlist):
      msg = str("A" if i==1 else "B") + " " + str(x[0:3])
      print(msg)
      file.write(msg + "\n")

plot_pair_respondents(0, 1, labels, X, spec_series, all_nums)

plot_pair_items(20, 24, -1, 1, X, spec_series, all_nums)      
