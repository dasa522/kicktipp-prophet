from models.base import PredictionModel
import pandas as pd
import numpy as np
from scipy.stats import poisson
from scipy import optimize

class DixonColes(PredictionModel):
    """ """
    name = 'dixonColes'

    def __init__(self, time_decay_alpha =0.001, regularization_lambda= 0.01, max_goals = 12):
        super().__init__() # Correctly call the base constructor
        self.max_goals = max_goals
        self.attack_params = None
        self.defense_params = None
        self.home_advantage = None
        self.rho = None
        self.time_decay_alpha = time_decay_alpha
        self.regularization_lambda = regularization_lambda # L2 penalty strength

    def tau(self,i,j,lam, mu, rho):
        # This function now ensures it never returns a negative value
        val = np.ones_like(lam,dtype=float)
        mask_0_0 = (i == 0) & (j == 0)
        mask_1_0 = (i == 1) & (j == 0)
        mask_0_1 = (i == 0) & (j == 1)
        mask_1_1 = (i == 1) & (j == 1)
        val[mask_0_0] = 1- (lam[mask_0_0] * mu[mask_0_0] * rho)
        val[mask_1_0] = 1 + (mu[mask_1_0] * rho)
        val[mask_0_1] = 1 + (lam[mask_0_1] * rho)
        val[mask_1_1] = 1 - rho
        
        # Clip the value to ensure it's not negative
        return np.maximum(1e-9, val)
    
    def fit(self, df: pd.DataFrame):
        df = df.copy()

        #get unique teamNames
        names = pd.concat([df['HomeTeam'],df['AwayTeam'] ]).unique()
        names.sort()
        # Set self.teams so that _normalize_team works correctly
        self.teams = pd.DataFrame(index=names)
        names_map = {team: i for i,team in enumerate(names)}
        number_teams = len(names)
        #add time decay weights to reduce impact of games in the far past
        df['Date'] = pd.to_datetime(df['Date'])
        df['age_in_days'] = (df['Date'].max() - df['Date']).dt.days
        weights = np.exp(-0.001 * df['age_in_days'].values) 
        
        # --- Pre-calculate indices for vectorization ---
        home_indices = df['HomeTeam'].map(names_map).values
        away_indices = df['AwayTeam'].map(names_map).values
        home_goals = df['FTHG'].values
        away_goals = df['FTAG'].values
        
        initial_params = np.concatenate((
            np.ones(number_teams), #attack 
            np.ones(number_teams), #defense
            [1.0],
            [0.0]
        ))


        def obj_func(x):
            attack_param = x[0:number_teams]
            defense_param = x[number_teams:2*number_teams]
            home_adv_param = x[2*number_teams]
            rho = x[2*number_teams+1]

            # --- Vectorized Calculation (replaces the for loop) ---
            # Calculate lambdas for all matches at once
            lambda_home = attack_param[home_indices] * defense_param[away_indices] * home_adv_param
            lambda_away = attack_param[away_indices] * defense_param[home_indices]
            
            # Calculate probability for all matches at once
            p = self.tau(home_goals, away_goals, lambda_home, lambda_away, rho) * \
                poisson.pmf(home_goals, lambda_home) * \
                poisson.pmf(away_goals, lambda_away)
            
            # Sum the log-likelihood across all matches
            log_likelihood = np.sum(weights * np.log(p + 1e-9))

            # Penalize the sum of squares of the parameters to prevent them from getting too large
            l2_penalty = self.regularization_lambda * np.sum(x**2)

            return -log_likelihood + l2_penalty

        # Bounds: attack, defense, and home_adv must be positive. rho is unbounded.
        bounds = [(0.0001, None)] * (2 * number_teams + 1) + [(None, None)]

        # Constraint: The sum of attack parameters must equal the number of teams
        constraints = [{'type': 'eq', 'fun': lambda x: sum(x[0:number_teams]) - number_teams}]
        optimize_res :optimize.OptimizeResult = optimize.minimize(obj_func, initial_params,bounds=bounds, constraints=constraints)
        
        if not optimize_res.success:
            print("Warning: Optimizer failed to converge.")
            return 
        param_list = optimize_res.x
        # --- Unpack and store the optimized parameters ---
        self.attack_params = pd.Series(param_list[0:number_teams], index=names)
        self.defense_params = pd.Series(param_list[number_teams:2*number_teams], index=names)
        self.home_advantage = param_list[2*number_teams]
        self.rho = param_list[2*number_teams+1]


    def predict(self, home_team, away_team):
        h_team = self._normalize_team(home_team)
        a_team = self._normalize_team(away_team)
        lam_home = self.attack_params[h_team]*self.defense_params[a_team] * self.home_advantage
        lam_away = self.attack_params[a_team]*self.defense_params[h_team] 

        best = (0, 0)
        best_p = 0.0
        for i in range(self.max_goals + 1):
            for j in range(self.max_goals + 1):
                p = self.tau(i,j, lam_home, lam_away, self.rho) * poisson.pmf(i, lam_home) * poisson.pmf(j, lam_away)
                if p > best_p:
                    best_p = p
                    best = (i, j)
        return best
