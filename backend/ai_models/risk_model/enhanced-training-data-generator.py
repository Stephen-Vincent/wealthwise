"""
Complete Risk Assessment Dataset Generator
Generates a comprehensive training dataset with full 1-100 risk score range
"""

import pandas as pd
import numpy as np
import random
import os
from pathlib import Path

class ComprehensiveRiskDataGenerator:
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Define all response options
        self.loss_tolerance_options = ['sell_immediately', 'wait_and_see', 'buy_more']
        self.panic_behavior_options = ['yes_always', 'yes_sometimes', 'no_never', 'no_experience']
        self.financial_behavior_options = ['invest_all', 'save_half', 'save_all', 'spend_it']
        self.engagement_level_options = ['daily', 'weekly', 'monthly', 'quarterly', 'rarely']
        self.investment_goal_options = ['buy a house', 'vacation', 'emergency fund', 'retirement', 'save for a car', 'wealth building']
        self.income_options = ['low', 'medium', 'high']
        
        # Enhanced scoring system for full range
        self.loss_tolerance_scores = {
            'sell_immediately': 8,   # Very conservative
            'wait_and_see': 25,      # Moderate
            'buy_more': 40          # Aggressive
        }
        
        self.panic_behavior_scores = {
            'yes_always': 5,         # Panic seller
            'yes_sometimes': 15,     # Sometimes emotional
            'no_never': 35,          # Steady investor
            'no_experience': 20      # Unknown/moderate
        }
        
        self.financial_behavior_scores = {
            'spend_it': 8,           # Present-focused
            'save_all': 12,          # Very conservative
            'save_half': 22,         # Balanced
            'invest_all': 35         # Growth-oriented
        }
        
        self.engagement_scores = {
            'rarely': 8,             # Passive
            'quarterly': 15,         # Low engagement
            'monthly': 25,           # Moderate
            'weekly': 30,            # Active
            'daily': 20              # Overactive (can hurt returns)
        }
        
        # Goal-based risk multipliers (enhanced)
        self.goal_risk_multipliers = {
            'emergency fund': 0.3,    # Very conservative
            'vacation': 0.6,          # Short-term, moderate
            'save for a car': 0.7,    # Medium-term
            'buy a house': 0.8,       # Important goal, moderate risk
            'wealth building': 1.4,   # Can be aggressive
            'retirement': 1.6         # Long-term, can take high risk
        }
        
        # Income capacity multipliers (enhanced)
        self.income_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.5
        }
    
    def experience_score(self, years):
        """Enhanced experience scoring for wider range."""
        if years == 0:
            return 5
        elif years <= 2:
            return 10
        elif years <= 5:
            return 18
        elif years <= 10:
            return 28
        elif years <= 20:
            return 35
        else:
            return 40  # Very experienced
    
    def timeframe_multiplier(self, years):
        """Timeframe affects risk capacity."""
        if years <= 1:
            return 0.5   # Very short term
        elif years <= 3:
            return 0.7   # Short term
        elif years <= 10:
            return 1.0   # Medium term
        elif years <= 20:
            return 1.3   # Long term
        else:
            return 1.6   # Very long term
    
    def calculate_comprehensive_risk_score(self, row):
        """
        Calculate risk score with enhanced methodology for full 1-100 range.
        """
        
        # Behavioral Risk Component (50% weight)
        behavioral_score = (
            self.loss_tolerance_scores[row['loss_tolerance']] * 0.35 +      # 35%
            self.panic_behavior_scores[row['panic_behavior']] * 0.25 +      # 25%
            self.financial_behavior_scores[row['financial_behavior']] * 0.25 + # 25%
            self.engagement_scores[row['engagement_level']] * 0.15          # 15%
        )
        
        # Experience Component (20% weight)
        experience_component = self.experience_score(row['years_of_experience'])
        
        # Financial Capacity Component (30% weight)
        income_score = {'low': 15, 'medium': 30, 'high': 45}[row['income']]
        
        # Investment capacity relative to target
        total_investment = row['lump_sum_investment'] + (row['monthly_investment'] * 12 * row['timeframe'])
        if row['target_amount'] > 0:
            capacity_ratio = min(total_investment / row['target_amount'], 3.0)
            investment_capacity_score = capacity_ratio * 20  # 0-60 points
        else:
            investment_capacity_score = 10
        
        # Timeframe capacity
        timeframe_component = self.timeframe_multiplier(row['timeframe']) * 25
        
        # Combine all components
        base_score = (
            behavioral_score * 0.40 +           # 40% behavioral
            experience_component * 0.25 +       # 25% experience  
            income_score * 0.15 +               # 15% income
            investment_capacity_score * 0.10 +  # 10% investment capacity
            timeframe_component * 0.10          # 10% timeframe
        )
        
        # Apply goal-based multiplier
        goal_adjusted_score = base_score * self.goal_risk_multipliers[row['investment_goal']]
        
        # Apply income multiplier for final capacity adjustment
        income_adjusted_score = goal_adjusted_score * self.income_multipliers[row['income']]
        
        # Add controlled randomness for realism
        noise = np.random.normal(0, 4)  # ±4 point standard deviation
        final_score = income_adjusted_score + noise
        
        # Ensure we use the full 1-100 range
        return max(1, min(100, final_score))
    
    def generate_targeted_samples(self, target_risk_range, num_samples):
        """Generate samples targeting specific risk ranges."""
        samples = []
        
        for _ in range(num_samples):
            if target_risk_range == 'ultra_conservative':  # 1-20
                # Ultra conservative profiles
                years_exp = random.choice([0, 1, 2])
                loss_tolerance = random.choice(['sell_immediately', 'wait_and_see'])
                panic_behavior = random.choice(['yes_always', 'yes_sometimes'])
                financial_behavior = random.choice(['save_all', 'spend_it'])
                engagement = random.choice(['rarely', 'quarterly'])
                goal = random.choice(['emergency fund', 'vacation'])
                income = random.choice(['low', 'medium'])
                timeframe = random.choice([1, 2, 3])
                
            elif target_risk_range == 'conservative':  # 20-40
                years_exp = random.choice([1, 2, 3, 4, 5])
                loss_tolerance = random.choice(['sell_immediately', 'wait_and_see'])
                panic_behavior = random.choice(['yes_sometimes', 'no_experience'])
                financial_behavior = random.choice(['save_all', 'save_half'])
                engagement = random.choice(['rarely', 'quarterly', 'monthly'])
                goal = random.choice(['emergency fund', 'save for a car', 'buy a house'])
                income = random.choice(['low', 'medium', 'high'])
                timeframe = random.choice([2, 3, 4, 5])
                
            elif target_risk_range == 'moderate':  # 40-60
                years_exp = random.choice([3, 4, 5, 6, 7, 8])
                loss_tolerance = random.choice(['wait_and_see', 'buy_more'])
                panic_behavior = random.choice(['yes_sometimes', 'no_never', 'no_experience'])
                financial_behavior = random.choice(['save_half', 'invest_all'])
                engagement = random.choice(['monthly', 'quarterly', 'weekly'])
                goal = random.choice(['buy a house', 'wealth building', 'retirement'])
                income = random.choice(['medium', 'high'])
                timeframe = random.choice([5, 7, 10, 12])
                
            elif target_risk_range == 'aggressive':  # 60-80
                years_exp = random.choice([8, 10, 12, 15, 18])
                loss_tolerance = random.choice(['buy_more', 'wait_and_see'])
                panic_behavior = random.choice(['no_never', 'yes_sometimes'])
                financial_behavior = random.choice(['invest_all', 'save_half'])
                engagement = random.choice(['weekly', 'monthly', 'daily'])
                goal = random.choice(['wealth building', 'retirement'])
                income = random.choice(['medium', 'high'])
                timeframe = random.choice([10, 15, 20])
                
            else:  # ultra_aggressive: 80-100
                years_exp = random.choice([15, 20, 25, 30])
                loss_tolerance = 'buy_more'
                panic_behavior = random.choice(['no_never', 'yes_sometimes'])
                financial_behavior = 'invest_all'
                engagement = random.choice(['daily', 'weekly', 'monthly'])
                goal = random.choice(['retirement', 'wealth building'])
                income = random.choice(['high', 'medium'])
                timeframe = random.choice([15, 20, 25, 30])
            
            # Generate financial amounts based on profile
            target_amount = self.generate_target_amount(goal, income, timeframe)
            lump_sum, monthly = self.generate_investment_amounts(
                target_amount, income, timeframe, years_exp
            )
            
            sample = {
                'years_of_experience': years_exp,
                'loss_tolerance': loss_tolerance,
                'panic_behavior': panic_behavior,
                'financial_behavior': financial_behavior,
                'engagement_level': engagement,
                'investment_goal': goal,
                'target_amount': target_amount,
                'lump_sum_investment': lump_sum,
                'monthly_investment': monthly,
                'timeframe': timeframe,
                'income': income
            }
            
            # Calculate risk score
            sample['risk_score'] = round(self.calculate_comprehensive_risk_score(sample), 1)
            samples.append(sample)
        
        return samples
    
    def generate_target_amount(self, goal, income, timeframe):
        """Generate realistic target amounts."""
        base_amounts = {
            'emergency fund': {'low': (3000, 20000), 'medium': (5000, 35000), 'high': (10000, 60000)},
            'vacation': {'low': (2000, 12000), 'medium': (3000, 20000), 'high': (5000, 40000)},
            'save for a car': {'low': (10000, 35000), 'medium': (15000, 55000), 'high': (25000, 100000)},
            'buy a house': {'low': (30000, 100000), 'medium': (50000, 200000), 'high': (80000, 500000)},
            'wealth building': {'low': (20000, 100000), 'medium': (50000, 300000), 'high': (100000, 1000000)},
            'retirement': {'low': (100000, 500000), 'medium': (200000, 1000000), 'high': (500000, 3000000)}
        }
        
        min_amt, max_amt = base_amounts[goal][income]
        
        # Adjust for timeframe
        if timeframe >= 20:
            max_amt *= 1.5
        elif timeframe >= 10:
            max_amt *= 1.2
        
        return int(np.random.uniform(min_amt, max_amt))
    
    def generate_investment_amounts(self, target_amount, income, timeframe, experience):
        """Generate realistic investment amounts."""
        # Base capacity ratios
        capacity_ratios = {
            'low': (0.4, 1.2),
            'medium': (0.6, 1.8),
            'high': (1.0, 3.0)
        }
        
        min_ratio, max_ratio = capacity_ratios[income]
        
        # Experience affects capacity
        exp_multiplier = 1.0 + (experience * 0.02)  # Up to 60% bonus for 30 years exp
        
        total_capacity = target_amount * random.uniform(min_ratio, max_ratio) * exp_multiplier
        
        # Split between lump sum and monthly
        lump_sum_preference = random.uniform(0.1, 0.8)
        lump_sum = int(total_capacity * lump_sum_preference)
        
        remaining_capacity = total_capacity - lump_sum
        monthly = int(remaining_capacity / (timeframe * 12)) if timeframe > 0 else 0
        
        return max(0, lump_sum), max(0, monthly)
    
    def generate_complete_dataset(self, total_samples=2000, output_file='backend/ai_models/training_data/complete_risk_dataset.csv'):
        """Generate comprehensive dataset with full risk range coverage."""
        
        print(f"🔄 Generating comprehensive risk dataset with {total_samples} samples...")
        
        all_samples = []
        
        # Target distribution across risk ranges
        distributions = {
            'ultra_conservative': int(total_samples * 0.15),  # 15%
            'conservative': int(total_samples * 0.25),        # 25%
            'moderate': int(total_samples * 0.30),            # 30%
            'aggressive': int(total_samples * 0.20),          # 20%
            'ultra_aggressive': int(total_samples * 0.10)     # 10%
        }
        
        for risk_range, num_samples in distributions.items():
            print(f"   📊 Generating {num_samples} {risk_range} samples...")
            samples = self.generate_targeted_samples(risk_range, num_samples)
            all_samples.extend(samples)
        
        # Convert to DataFrame and shuffle
        df = pd.DataFrame(all_samples)
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Display statistics
        print(f"\n📈 Dataset Statistics:")
        print(f"   Total samples: {len(df)}")
        print(f"   Risk score range: {df['risk_score'].min():.1f} - {df['risk_score'].max():.1f}")
        print(f"   Mean risk score: {df['risk_score'].mean():.1f}")
        print(f"   Standard deviation: {df['risk_score'].std():.1f}")
        
        # Risk distribution
        ultra_conservative = len(df[df['risk_score'] < 20])
        conservative = len(df[(df['risk_score'] >= 20) & (df['risk_score'] < 40)])
        moderate = len(df[(df['risk_score'] >= 40) & (df['risk_score'] < 60)])
        aggressive = len(df[(df['risk_score'] >= 60) & (df['risk_score'] < 80)])
        ultra_aggressive = len(df[df['risk_score'] >= 80])
        
        print(f"\n🎯 Risk Distribution:")
        print(f"   Ultra Conservative (1-20): {ultra_conservative} ({ultra_conservative/len(df)*100:.1f}%)")
        print(f"   Conservative (20-40): {conservative} ({conservative/len(df)*100:.1f}%)")
        print(f"   Moderate (40-60): {moderate} ({moderate/len(df)*100:.1f}%)")
        print(f"   Aggressive (60-80): {aggressive} ({aggressive/len(df)*100:.1f}%)")
        print(f"   Ultra Aggressive (80-100): {ultra_aggressive} ({ultra_aggressive/len(df)*100:.1f}%)")
        
        # Save dataset
        df.to_csv(output_file, index=False)
        print(f"\n✅ Complete risk dataset saved to: {output_file}")
        
        return df

def main():
    """Generate the complete risk assessment dataset."""
    
    # Set up paths
    script_dir = Path(__file__).parent
    output_file = script_dir / "training_data" / "complete_risk_dataset.csv"
    
    # Create training_data directory if it doesn't exist
    output_file.parent.mkdir(exist_ok=True)
    
    # Generate dataset
    generator = ComprehensiveRiskDataGenerator()
    df = generator.generate_complete_dataset(
        total_samples=2000, 
        output_file=str(output_file)
    )
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"📁 File location: {output_file}")
    print(f"📊 Ready to train your enhanced risk model!")
    
    # Show sample data
    print(f"\nSample data preview:")
    print(df.head(3).to_string())

if __name__ == "__main__":
    main()