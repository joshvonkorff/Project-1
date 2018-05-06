## Modernization and democracy

This project is based on work by Ronald Inglehart and Christian Welzel.  They were interested in understanding how to promote democracy throughout the world; in particular, they believed that modernization leads to democracy.  They wrote a book on this topic called "Modernization, Cultural Change, and Democracy" about their survey, called the "World Values Survey."

Based on the survey, Inglehart proposed that countries can be classified on two axes.  The first axis, which is called the "traditional vs. secular-rational values axis", has to do with religiosity and related values.  The second axis, which is called the "survival vs. self-expression values axis" is supposed to be relate to a culture of democracy as well as to modernization.  Inglehart believes that democracy will come with material wealth via this self-expression axis: as people become wealthier, they can afford to be more tolerant, for example.

## My hypothesis

I was somewhat skeptical of this claim.  I believe Inglehart when he says that democracy and wealth are related to some degree, but on the other hand, American democracy dates back to the late 1700s when America was quite poor.  This suggests that modernization and democracy may in some cases be quite unrelated.  I decided to check Inglehart's data, available on:

http://www.worldvaluessurvey.org/wvs.jsp

In particular, I used the "Wave 6: 2010-2014" data located at:

http://www.worldvaluessurvey.org/WVSDocumentationWV6.jsp

My goal was to take Inglehart's survey items, which he classifies as traditional/secular and survival/self-expression, and to try to cluster them into two clusters (hopefully reproducing Inglehart's two groups) as well as four clusters, to see if I could separate the different aspects of the survival/self-expression axis.  I hypothesized that the "material wealth" items would come out as a separate cluster if I did that.

## Clustering analysis

I used a clustering analysis method called Affinity Propagation from the scikit-learn library.  I analyzed data at the level of the United States (about 2,000 respondents) and at the level of the whole world (about 90,000 respondents.)

One challenge was that the survey items could be correlated or anti-correlated.  For instance, the following two items are closely related but anti-correlated:

* {Would not like to have as neighbors: Heavy drinkers}
* {Important child qualities: Tolerance and respect for other people}

Unsurprisingly, people who devalue tolerance as a quality in children are themselves less likely to tolerate people who they perceive as objectionable.  The puzzle was how to cluster these items together although they have polar opposite values.

The solution was that Affinity Propagation permits the user to input a "precomputed" affinity matrix showing all possible user-defined distances between pairs of survey items.  I then defined the distance to be the minimum of (1) the actual distance between the survey items, (2) the distance between one survey item and the negative of the other.  (Survey items had to be converted to Z scores in order for any comparison to make sense.)

## Results

The result confirmed my suspicion.  When clustering the U.S. data into two clusters, they seemed to fit Inglehart's distinction moderately well: about 75% of the items in one cluster were traditional/secular and 75% in the other were survival/self-expression.  However, when splitting the clusters into four, it was clear that the items related to material prosperity were grouped together and not with democracy.

These items were:
* {Important in life: Friends}
* {Important in life: Leisure time}
* {Feeling of happiness}
* {State of health (subjective)}
* {How much freedom of choice and control over own life}
* {Satisfaction with financial situation of household}

The items clustered with democracy were:

* {Most people can be trusted}
* {Interest in politics}
* {Private vs state ownership of business}
* {Importance of democracy}
* {Post-materialist index (4-item)}

Thus, people were more likely to approve of democracy if they were trusting, interested in politics, and approved of state ownership of business.  (You can't tell which items are correlated vs. anticorrelated without consulting the specification/readme file and manually examining each item one at a time.  This is because the specification file is written in a fairly unstandardized way that might require much more work to decode automatically.  However, I manually checked that it was state ownership, not corporate ownership, that was associated with democracy.)

At a global scale, the correlates and anticorrelates of democracy were quite different.  There, they were:

* {Would not like to have as neighbors: Immigrants/foreign workers}
* {When jobs are scarce, men should have more right to a job than women}
* {If a woman earns more money than her husband, it's almost certain to cause problems}
* {On the whole, men make better political leaders than women do}
* {A university education is more important for a boy than for a girl}
* {Being a housewife is just as fulfilling as working for pay}
* {Political system: Having a strong leader who does not have to bother with parliament and elections}
* {Importance of democracy}

Thus, it seems that more modern views about women's roles are especially common in democratic countries.

## Extended printout

The following is the complete printout of the results for the four-cluster analysis of the United States.  An explanation of the output format is given within the output itself.

```
The number of good survey items for the United States is:  31
The number of respondents for the United States is:  2232
How to interpret these resuls:

Clustering was performed by Affinity Propagation.
The number 0, 1, 2, or 3 is the cluster number.  All 0's are in the same cluster.
Surv-Exp means that this item is classified as Survival vs. Self-Expression by Inglehart.
Trad-Sec means that this item is classified as Traditional vs. Secular by Inglehart.
The phrase in brackets { } describes the nature of the item.

A's are correlated with other A's and anticorrelated with B's in that cluster.
However, to interpret this you need to know whether the scale on that item runs from e.g.
1 (highest) to 5 (lowest) or the reverse.  This requires manually consulting a table.

A (0, 'Trad-Sec', '{Important in life: Family}')
A (0, 'Trad-Sec', '{Important in life: Work}')
A (0, 'Trad-Sec', '{Important in life: Religion}')
B (0, 'Surv-Exp', '{Important child qualities: Imagination}')
B (0, 'Trad-Sec', '{Active/Inactive membership: Church or religious organization}')
A (0, 'Trad-Sec', '{One of my main goals in life has been to make my parents proud}')
A (0, 'Surv-Exp', '{Being a housewife is just as fulfilling as working for pay}')
A (0, 'Trad-Sec', '{Confidence: The Churches}')
A (0, 'Trad-Sec', '{Justifiable: Divorce}')
A (0, 'Trad-Sec', '{Justifiable: Suicide}')
A (0, 'Trad-Sec', '{Autonomy Index}')
A (0, 'Trad-Sec', '{Overall Secular Values-1: Inverse respect for authority}')
A (0, 'Trad-Sec', '{Overall Secular Values-1: Inverse national pride}')
A (1, 'Surv-Exp', '{Important in life: Friends}')
A (1, 'Surv-Exp', '{Important in life: Leisure time}')
A (1, 'Surv-Exp', '{Feeling of happiness}')
A (1, 'Surv-Exp', '{State of health (subjective)}')
B (1, 'Surv-Exp', '{How much freedom of choice and control over own life}')
B (1, 'Surv-Exp', '{Satisfaction with financial situation of household}')
B (2, 'Surv-Exp', '{Important child qualities: Hard work}')
A (2, 'Surv-Exp', '{Important child qualities: Tolerance and respect for other people}')
B (2, 'Surv-Exp', '{When jobs are scarce, men should have more right to a job than women}')
B (2, 'Trad-Sec', "{If a woman earns more money than her husband, it's almost certain to cause problems}")
B (2, 'Surv-Exp', '{On the whole, men make better political leaders than women do}')
B (2, 'Surv-Exp', '{A university education is more important for a boy than for a girl}')
A (2, 'Surv-Exp', '{Government responsibility}')
A (3, 'Surv-Exp', '{Most people can be trusted}')
A (3, 'Trad-Sec', '{Interest in politics}')
A (3, 'Surv-Exp', '{Private vs state ownership of business}')
B (3, 'Surv-Exp', '{Importance of democracy}')
B (3, 'Surv-Exp', '{Post-materialist index (4-item)}')
```