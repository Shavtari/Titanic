import pandas as pd

def get_score(wait=False):
    """
    Calculates score of prediction comparing it to truth.
    Score is given in float in range 0-1, where 0 is 0% accuracy and 1 is 100% accuracy.

    :param wait: bool, default=False.
        Setting to True makes it wait for user input after running.
    :return: score: float
    """
    pred = pd.read_csv("titanic_prediction.csv")
    truth = pd.read_csv('test_truth.csv')
    select_truth = truth[['PassengerId', 'Survived']]
    score = 0
    perfect_score = len(select_truth)

    for line in range(perfect_score):
        if select_truth.iloc[line, 1]==pred.iloc[line, 1]:
            score += 1

    print(f"\nYour score is: {score/perfect_score}")
    if wait:
        input('Press Enter to exit...')
    return score/perfect_score

if __name__ == "__main__":
    #Gives score when file is opened as a script, waiting for user input before closing console
    get_score(wait=True)