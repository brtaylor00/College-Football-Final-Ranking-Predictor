# Predicting College Football Rankings and Playoff Outcomes

## Research Question
How well can preseason information be used to predict a college football team's final AP ranking and whether it makes the College Football Playoff?

## Background and Context
College football writers vote every week of the season, from the preseason to the end of the season, on who the top 25 teams in the country are, and the votes are added up to produce a top 25 ranking. The main poll for the season is the AP Poll, which has been running since 1936. In the preseason, it attempts to rank the 25 teams based on little information, such as the previous season's record, name brand, recruiting, and the transfer portal. But this ranking often produces many teams that end up struggling, and fail to remain ranked by the end of the season. At the end of the season, 12 teams, generally the 12 best teams in the country, make it to the College Football Playoff, and one final AP ranking comes out following the playoffs. Every season includes several preseason ranked teams who do not stay ranked, and also postseason ranked teams that did not begin ranked. So how can we find a correlation to how the preseason ranking and final ranking, or more particularly, whether they make the College Football Playoff?
This project is a machine learning project to find a regression problem that predicts a team's final Associated Press (AP) ranking at the end of the season. The second is a classification problem that predicts whether a team makes the College Football Playoff (CFP).

## Dataset
Due to the College Football Playoffs being founded in 2014, the dataset will include all teams from 2014 to the most recently completed season, 2025. We can also look at the 2026 preseason poll to determine teams odds at making the CFP.
The dataset, found from Wikipedia, includes the preseason AP poll and final AP poll from each season, and teams to make the College Football Playoff. It also includes the previous season's final ranking for each team (so the 2014 rankings uses the 2013 final poll).
The initial dataset contains 430 team-season observations. Because the regression target requires a final AP ranking, the regression analysis uses 300 observations for teams that received a final AP ranking. The classification analysis uses the larger 430-observation dataset.
The dataset contains some missing AP rankings because not every team was ranked in the AP Top 25 at the beginning or end of a season. These missing rankings were handled during the data-preparation process rather than simply removing those observations.

## Variables
The variables used in this project were:
- preseason_ap
- previous_final_ap
- final_ap_rank
- made_playoff

Preseason_AP: this represents a teams initial ranking in the preseason poll, before any games have taken place, for a given season. A lower value represents a better ranking (1 is the best, 25 is the worst. However, due to only 25 out of over 120 teams being ranked, 25 is still very good). Unranked teams were assigned a value of 26.
Previous_Final_AP: this represents a teams final ranking from the prior season. So 2017 Alabama will use the final ranking of 2016 Alabama. Again, unranked teams were assigned a value of 26.
Final_AP_Rank: this represents a teams final ranking from the given season, from the same season that the preseason ranking is used. This is the target variable for the regression problem.
Made_playoff: This is a binary to represent whether a team made the College Football Playoff. A value of 1 indicates that they made the playoffs, and a value of 0 indicates that they did not. From 2014 to 2023, only 4 teams made the playoffs, and from 2024 to 2025, 12 teams made it. This is a total of 64 teams to make the CFP. This is the target variable for the classification problem.

## Data Understanding and Exploration
Before developing the machine-learning models, I explored the distribution of the variables and examined how preseason expectations and previous-season performance related to the eventual outcomes.

The classification dataset contains 430 team-season observations. Of these observations, 64 teams (14.9%) made the College Football Playoff, while 366 teams (85.1%) did not. This represents a substantial class imbalance. Because relatively few observations belong to the playoff class, accuracy alone would not be a sufficient measure of classification performance. This influenced my decision to also use precision, recall, F1 score, and ROC-AUC when evaluating the classification models.

The AP ranking variables also showed an important characteristic: lower ranking numbers represent better rankings. For example, a preseason ranking of 1 indicates a team was ranked first, while a value of 25 indicates the team was ranked 25th. Teams that were not ranked in the AP Top 25 were represented by a value of 26. This allowed unranked teams to remain in the analysis rather than being removed.

The data also suggest a relationship between preseason expectations and final performance. Teams with stronger preseason AP rankings generally tended to finish with stronger final AP rankings. Previous-season AP rankings provided additional information about a team's recent performance, although their relationship with the final ranking appeared weaker than the preseason ranking.

For the classification problem, the same pattern was expected: teams receiving stronger preseason rankings generally had a greater likelihood of reaching the College Football Playoff. However, preseason rankings were not perfect predictors, since teams can substantially outperform or underperform expectations during a season. Teams who started a season unranked still occasionally made the playoffs, especially in the expanded 2024 and 2025 playoffs.

I used visualizations to examine these relationships before modeling. These included distributions of AP rankings, comparisons of preseason and final rankings, and visualizations of the playoff outcome. The exploration showed that the dataset was not evenly distributed across the classification target and that preseason AP ranking was likely to be an important predictor.

These findings informed the feature-selection process. I retained preseason AP ranking and previous-season final AP ranking because both variables are available before the season's final outcome and provide information about preseason expectations and recent performance. Variables that would only become available after the season began were excluded because using them would introduce information that would not actually be available when making a preseason prediction.

## Data Cleaning and Preparation
The data was cleaned and prepared before training the machine-learning models. Team names were standardized so that teams could be matched across seasons, and duplicate team-season observations were checked. For example, searching for Alabama could show South Alabama as well, so they had to be split. No duplicate observations were found.

Because AP rankings are only published for the Top 25 teams, some teams did not have a preseason or previous-season final ranking. Rather than removing these observations, I represented an unranked team with a value of 26, indicating that it was outside the AP Top 25. This allowed unranked teams to remain in the analysis. While unranked teams still can receive votes to be ranked, they still are unranked, even when they are ranked 26th in the poll, just one spot away from being ranked.

For the regression analysis, observations without a final AP ranking were excluded because a final numerical ranking was required as the target. The classification analysis retained the larger dataset because the playoff outcome could still be identified.

## Baseline and Model Development
I established a baseline for each prediction task before training the machine-learning models. For the regression problem, the baseline predicted the mean final AP ranking for every team. For the classification problem, the baseline always predicted the most common outcome, which was that a team would not make the College Football Playoff. These baselines provide simple reference points for determining whether the machine-learning models provide useful improvement.

For the regression task, I compared Linear Regression and Random Forest Regression. Linear Regression was selected because it provides a simple and interpretable way to model the relationship between the AP predictors and final ranking. Random Forest was included as a more flexible model capable of capturing nonlinear relationships.

For the classification task, I compared Logistic Regression and Random Forest Classification. Logistic Regression was appropriate because the target is a binary outcome, while Random Forest provided a nonlinear alternative.

The models were trained using the same predictors and chronological training and validation data so that their performance could be compared fairly. I used a minimum leaf size of 3 and 300 trees for the Random Forest models. The classification models also used balanced class weights because only 14.9% of the observations represented playoff teams.

Model selection was based on performance on the 2023-2024 validation period rather than the final 2025 test set. This prevented the test season from influencing the choice of model.

## Model Evaluation and Selection
Different evaluation metrics were used for the two prediction tasks. For regression, I used Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R². MAE represents the average number of ranking positions by which predictions differ from the actual result, while RMSE gives greater weight to larger errors. R² measures how much of the variation in final AP ranking is explained by the model.

For classification, I used accuracy, precision, recall, F1 score, and ROC-AUC. Because playoff teams represented only 14.9% of the dataset, accuracy alone could be misleading. Recall and F1 score were particularly important because they measure how effectively the model identifies playoff teams.

# Regression Results

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Mean Baseline | 6.240 | 7.211 | 0.000 |
| Linear Regression | 5.008 | 5.712 | 0.373 |
| Random Forest | 5.218 | 6.163 | 0.270 |

## Ethics and Limitations

## Conclusion

## Code and AI Transparency

## References
