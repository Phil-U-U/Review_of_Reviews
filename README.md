# Review_of_Reviews

Motivation:
The ratings from users who have consumed the products act as a good reference for future buyers. The current Amazon rating system considers the age of review, helpful votes and whether the reviews come from verified perchases, but not the review quality. For those high quality reviews with detailed explanation of why the user likes or dislikes the product should have higher weight than those reviews with few words. On the other hand, it is nice to put high quality reviews on top of low quality ones, for the convenience of potential buyers. The helpful votes ratio is a good indicator of review quality. However, only about 12 percent of Amazon book reviews have more than 8 votes. The majority of them, either have no vote at all, or only few votes, which doesn't tell much how helpful the reviews are. 
Based on the above considerations, we use the ration of helpful votes, i.e., number of people who found the reviews are helpful / total number of votes for the product, as the metric to evaluate the quality of reviews (helpfulness score), and design machine learning model to predict the helpfulness score. 


Here is our model:


