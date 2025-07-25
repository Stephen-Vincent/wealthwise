"""
Complete Risk Assessment Dataset Generator
Generates a comprehensive training dataset with full 1-100 risk score range
Updated with improved scoring methodology for better AI training
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
        
        # IMPROVED scoring system for BALANCED AI training
        self.loss_tolerance_scores = {
            'sell_immediately': 5,   # Very conservative
            'wait_and_see': 20,      # Moderate 
            'buy_more': 35          # Aggressive (reduced from 55)
        }
        
        self.panic_behavior_scores = {
            'yes_always': 0,         # Panic seller
            'yes_sometimes': 12,     # Sometimes emotional 
            'no_never': 30,          # Steady investor (reduced from 50)
            'no_experience': 15      # Unknown/moderate 
        }
        
        self.financial_behavior_scores = {
            'spend_it': 5,           # Present-focused
            'save_all': 10,          # Very conservative 
            'save_half': 20,         # Balanced 
            'invest_all': 35         # Growth-oriented (reduced from 60)
        }
        
        self.engagement_scores = {
            'rarely': 5,             # Passive 
            'quarterly': 12,         # Low engagement 
            'monthly': 20,           # Moderate 
            'weekly': 28,            # Active (reduced from 50)
            'daily': 25              # Overactive (slight penalty)
        }
        
        # BALANCED: Goal-based risk bonuses (reduced)
        self.goal_risk_bonuses = {
            'emergency fund': -15,    # Conservative penalty (reduced)
            'vacation': -8,           # Short-term penalty (reduced)
            'save for a car': 0,      # Neutral
            'buy a house': 8,         # Moderate bonus (reduced)
            'wealth building': 15,    # Growth bonus (reduced from 25)
            'retirement': 20          # Long-term bonus (reduced from 35)
        }
        
        # BALANCED: Income capacity bonuses (reduced)
        self.income_bonuses = {
            'low': -8,       # Penalty for limited capacity (reduced)
            'medium': 0,     # Neutral
            'high': 10       # Bonus for high capacity (reduced from 15)
        }
    
    def experience_score(self, years):
        """BALANCED experience scoring."""
        if years == 0:
            return 5
        elif years <= 2:
            return 10
        elif years <= 5:
            return 18
        elif years <= 10:
            return 25
        elif years <= 20:
            return 35
        else:
            return 40  # Very experienced (reduced from 70)
    
    def timeframe_multiplier(self, years):
        """BALANCED timeframe bonuses."""
        if years <= 1:
            return -10   # Very short term penalty (reduced)
        elif years <= 3:
            return -3    # Short term penalty (reduced)
        elif years <= 10:
            return 0     # Medium term neutral
        elif years <= 20:
            return 8     # Long term bonus (reduced from 15)
        else:
            return 15    # Very long term bonus (reduced from 25)
    
    def calculate_comprehensive_risk_score(self, row):
        """
        Calculate risk score with IMPROVED methodology for full 1-100 range.
        Uses additive scoring with strategic mapping for AI training.
        """
        
        # STEP 1: Core Behavioral Score (0-215 possible)
        behavioral_base = (
            self.loss_tolerance_scores[row['loss_tolerance']] +          # 0-55
            self.panic_behavior_scores[row['panic_behavior']] +          # 0-50
            self.financial_behavior_scores[row['financial_behavior']] +  # 0-60
            self.engagement_scores[row['engagement_level']]              # 0-50
        )
        
        # STEP 2: Add Experience Score (0-70)
        experience_score = self.experience_score(row['years_of_experience'])
        
        # STEP 3: Add Goal Bonus/Penalty (-20 to +35)
        goal_bonus = self.goal_risk_bonuses[row['investment_goal']]
        
        # STEP 4: Add Timeframe Bonus/Penalty (-15 to +25)
        timeframe_bonus = self.timeframe_multiplier(row['timeframe'])
        
        # STEP 5: Add Income Bonus/Penalty (-10 to +15)
        income_bonus = self.income_bonuses[row['income']]
        
        # STEP 6: Calculate Investment Capacity Bonus (0-30)
        total_investment = row['lump_sum_investment'] + (row['monthly_investment'] * 12 * row['timeframe'])
        if row['target_amount'] > 0:
            capacity_ratio = total_investment / row['target_amount']
            if capacity_ratio >= 2.0:
                capacity_bonus = 30      # Over-investing (very aggressive)
            elif capacity_ratio >= 1.5:
                capacity_bonus = 20      # Strong capacity
            elif capacity_ratio >= 1.0:
                capacity_bonus = 10      # Meeting target
            elif capacity_ratio >= 0.5:
                capacity_bonus = 5       # Limited capacity
            else:
                capacity_bonus = 0       # Minimal capacity
        else:
            capacity_bonus = 5
        
        # STEP 7: Combine All Components (Additive)
        raw_score = (
            behavioral_base +
            experience_score +
            goal_bonus +
            timeframe_bonus +
            income_bonus +
            capacity_bonus
        )
        
        # STEP 8: Add REDUCED Randomness
        noise = np.random.normal(0, 5)  # Reduced noise for more predictability
        noisy_score = raw_score + noise
        
        # STEP 9: MUCH MORE CONSERVATIVE MAPPING for Balanced Distribution
        # Map raw scores to ensure balanced distribution across 1-100
        if noisy_score < 40:
            # Map low scores to 1-25 (Ultra Conservative)
            mapped = 1 + (max(0, noisy_score) / 40) * 24
        elif noisy_score < 70:
            # Map medium-low scores to 25-45 (Conservative)
            mapped = 25 + ((noisy_score - 40) / 30) * 20
        elif noisy_score < 100:
            # Map medium scores to 45-65 (Moderate)
            mapped = 45 + ((noisy_score - 70) / 30) * 20
        elif noisy_score < 130:
            # Map medium-high scores to 65-80 (Moderate Aggressive)
            mapped = 65 + ((noisy_score - 100) / 30) * 15
        else:
            # Map high scores to 80-100 (Ultra Aggressive)
            mapped = 80 + min((noisy_score - 130) / 30, 1) * 20
        
        # Ensure bounds and return
        final_score = max(1, min(100, mapped))
        return final_score
    
    def generate_targeted_samples(self, target_risk_range, num_samples):
        """Generate samples targeting specific risk ranges with MUCH MORE CONSERVATIVE distributions."""
        samples = []
        
        for _ in range(num_samples):
            if target_risk_range == 'ultra_conservative':  # Target 1-25
                # VERY conservative profiles only
                years_exp = random.choice([0, 1, 2])
                loss_tolerance = random.choice(['sell_immediately', 'sell_immediately', 'wait_and_see'])  # Bias toward sell
                panic_behavior = random.choice(['yes_always', 'yes_sometimes'])
                financial_behavior = random.choice(['save_all', 'spend_it'])
                engagement = random.choice(['rarely', 'quarterly'])
                goal = random.choice(['emergency fund', 'vacation'])
                income = random.choice(['low', 'medium'])
                timeframe = random.choice([1, 2])
                
            elif target_risk_range == 'conservative':  # Target 25-45
                years_exp = random.choice([0, 1, 2, 3])
                loss_tolerance = random.choice(['sell_immediately', 'wait_and_see'])
                panic_behavior = random.choice(['yes_sometimes', 'no_experience'])
                financial_behavior = random.choice(['save_all', 'save_half'])
                engagement = random.choice(['rarely', 'quarterly'])
                goal = random.choice(['emergency fund', 'save for a car'])
                income = random.choice(['low', 'medium'])
                timeframe = random.choice([2, 3, 4])
                
            elif target_risk_range == 'moderate':  # Target 45-65
                years_exp = random.choice([3, 4, 5, 6])
                loss_tolerance = random.choice(['wait_and_see', 'wait_and_see', 'buy_more'])  # Bias toward wait
                panic_behavior = random.choice(['no_experience', 'yes_sometimes'])
                financial_behavior = random.choice(['save_half', 'save_half', 'invest_all'])  # Bias toward save_half
                engagement = random.choice(['monthly', 'quarterly'])
                goal = random.choice(['buy a house', 'save for a car'])
                income = random.choice(['medium', 'medium', 'high'])  # Bias toward medium
                timeframe = random.choice([5, 7, 8])
                
            elif target_risk_range == 'aggressive':  # Target 65-80
                years_exp = random.choice([6, 8, 10, 12])
                loss_tolerance = random.choice(['buy_more', 'wait_and_see'])
                panic_behavior = random.choice(['no_never', 'no_experience'])
                financial_behavior = random.choice(['invest_all', 'save_half'])
                engagement = random.choice(['weekly', 'monthly'])
                goal = random.choice(['wealth building', 'retirement'])
                income = random.choice(['medium', 'high'])
                timeframe = random.choice([10, 12, 15])
                
            else:  # ultra_aggressive: Target 80-100
                years_exp = random.choice([15, 20, 25])
                loss_tolerance = 'buy_more'
                panic_behavior = 'no_never'
                financial_behavior = 'invest_all'
                engagement = random.choice(['daily', 'weekly'])
                goal = random.choice(['retirement', 'wealth building'])
                income = 'high'
                timeframe = random.choice([20, 25, 30])
            
            # Generate MUCH MORE CONSERVATIVE financial amounts
            target_amount = self.generate_conservative_target_amount(goal, income, timeframe)
            lump_sum, monthly = self.generate_conservative_investment_amounts(
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
    
    def generate_conservative_target_amount(self, goal, income, timeframe):
        """Generate MORE CONSERVATIVE target amounts to avoid high capacity bonuses."""
        base_amounts = {
            'emergency fund': {'low': (3000, 15000), 'medium': (5000, 25000), 'high': (8000, 40000)},
            'vacation': {'low': (2000, 8000), 'medium': (3000, 15000), 'high': (5000, 25000)},
            'save for a car': {'low': (8000, 25000), 'medium': (12000, 40000), 'high': (20000, 60000)},
            'buy a house': {'low': (25000, 80000), 'medium': (40000, 150000), 'high': (60000, 300000)},
            'wealth building': {'low': (20000, 80000), 'medium': (40000, 200000), 'high': (80000, 500000)},
            'retirement': {'low': (80000, 400000), 'medium': (150000, 600000), 'high': (250000, 1000000)}
        }
        
        min_amt, max_amt = base_amounts[goal][income]
        
        # REDUCED timeframe multipliers
        if timeframe >= 20:
            max_amt *= 1.2  # Reduced from 1.8
        elif timeframe >= 10:
            max_amt *= 1.1  # Reduced from 1.4
        
        return int(np.random.uniform(min_amt, max_amt))
    
    def generate_conservative_investment_amounts(self, target_amount, income, timeframe, experience):
        """Generate MUCH MORE CONSERVATIVE investment amounts."""
        # MUCH MORE CONSERVATIVE capacity ratios
        capacity_ratios = {
            'low': (0.2, 0.6),      # Reduced from (0.3, 1.0)
            'medium': (0.3, 0.8),   # Reduced from (0.5, 1.5)
            'high': (0.4, 1.2)      # Reduced from (0.8, 2.5)
        }
        
        min_ratio, max_ratio = capacity_ratios[income]
        
        # REDUCED experience multiplier
        exp_multiplier = 1.0 + (experience * 0.01)  # Reduced from 0.03
        
        total_capacity = target_amount * random.uniform(min_ratio, max_ratio) * exp_multiplier
        
        # Split between lump sum and monthly
        lump_sum_pref = random.uniform(0.0, 0.7)  # Reduced from 0.9
        lump_sum = int(total_capacity * lump_sum_pref)
        
        remaining = total_capacity - lump_sum
        monthly = int(remaining / (timeframe * 12)) if timeframe > 0 else 0
        
        return max(0, lump_sum), max(0, monthly)
    
    def generate_complete_dataset(self, total_samples=3000, output_file='backend/ai_models/training_data/complete_risk_dataset.csv'):
        """Generate comprehensive dataset with full risk range coverage and IMPROVED distribution."""
        
        print(f"🔄 Generating comprehensive risk dataset with {total_samples} samples...")
        
        all_samples = []
        
        # IMPROVED distribution across risk ranges for better AI training
        distributions = {
            'ultra_conservative': int(total_samples * 0.10),  # 10% (reduced)
            'conservative': int(total_samples * 0.20),        # 20% (reduced)
            'moderate': int(total_samples * 0.40),            # 40% (increased)
            'aggressive': int(total_samples * 0.25),          # 25% (same)
            'ultra_aggressive': int(total_samples * 0.05)     # 5% (same)
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
        
        # Risk distribution with UPDATED ranges
        ultra_conservative = len(df[df['risk_score'] < 30])
        conservative = len(df[(df['risk_score'] >= 30) & (df['risk_score'] < 50)])
        moderate = len(df[(df['risk_score'] >= 50) & (df['risk_score'] < 70)])
        aggressive = len(df[(df['risk_score'] >= 70) & (df['risk_score'] < 85)])
        ultra_aggressive = len(df[df['risk_score'] >= 85])
        
        print(f"\n🎯 Risk Distribution:")
        print(f"   Ultra Conservative (1-30): {ultra_conservative} ({ultra_conservative/len(df)*100:.1f}%)")
        print(f"   Conservative (30-50): {conservative} ({conservative/len(df)*100:.1f}%)")
        print(f"   Moderate (50-70): {moderate} ({moderate/len(df)*100:.1f}%)")
        print(f"   Aggressive (70-85): {aggressive} ({aggressive/len(df)*100:.1f}%)")
        print(f"   Ultra Aggressive (85-100): {ultra_aggressive} ({ultra_aggressive/len(df)*100:.1f}%)")
        
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
        total_samples=3000,  # Increased for better AI training
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