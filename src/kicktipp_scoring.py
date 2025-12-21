def get_kicktipp_points(pred_home: int, pred_away: int, actual_home: int, actual_away: int) -> int:
    """
    Calculates Kicktipp points for a single prediction based on standard rules.
    
    - 4 points: Correct score
    - 3 points: Correct goal difference and winner
    - 2 points: Correct winner (or draw)
    - 0 points: Incorrect
    """
    # Exact score
    if pred_home == actual_home and pred_away == actual_away:
        return 4
    
    # Correct goal difference (and not an exact score)
    if (pred_home - pred_away) == (actual_home - actual_away):
        return 3
        
    # Correct winner/draw (and not correct difference)
    pred_outcome = 1 if pred_home > pred_away else (-1 if pred_home < pred_away else 0)
    actual_outcome = 1 if actual_home > actual_away else (-1 if actual_home < actual_away else 0)
    if pred_outcome == actual_outcome:
        return 2
        
    return 0