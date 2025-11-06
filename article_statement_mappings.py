#!/usr/bin/env python3
"""
Article-Statement Linkage Mappings for BCI Confirmation Bias Study
Defines which statements each article relates to and their bias types
Updated with correct relationships from study specification
"""

# Complete Article-Statement linkage map with correct relationships
# Format: article_code -> {type, headline, primary_statement, all_statements}
ARTICLE_STATEMENT_MAP = {
    # Topic T01: Climate Change and Environmental Policy
    'T01A': {
        'type': 'confirmatory',
        'headline': 'Carbon Pricing Drives Innovation',
        'statement': 'T01-S01',  # Primary statement for backward compatibility
        'weight': 1.0,
        'statements': {
            'T01-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T01-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T01B': {
        'type': 'disconfirmatory',
        'headline': 'Personal Choices Drive Climate Action',
        'statement': 'T01-S01',
        'weight': 1.0,
        'statements': {
            'T01-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T01-S05': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T01C': {
        'type': 'confirmatory',
        'headline': 'Renewable Energy Boom Creates Economic Windfall',
        'statement': 'T01-S02',
        'weight': 1.0,
        'statements': {
            'T01-S02': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T01-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T01D': {
        'type': 'disconfirmatory',
        'headline': 'Hidden Costs of Green Transition',
        'statement': 'T01-S02',
        'weight': 1.0,
        'statements': {
            'T01-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T01-S03': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T01E': {
        'type': 'neutral',
        'headline': 'Climate Policy Mixed Results',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T01-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T01-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T01-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T01-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T01-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T02: Technology and Social Media Impact
    'T02A': {
        'type': 'confirmatory',
        'headline': 'Teen Mental Health Crisis Linked to Social Media',
        'statement': 'T02-S01',
        'weight': 1.0,
        'statements': {
            'T02-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T02-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T02B': {
        'type': 'disconfirmatory',
        'headline': 'Digital Platforms Support Teen Wellbeing',
        'statement': 'T02-S01',
        'weight': 1.0,
        'statements': {
            'T02-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T02-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T02C': {
        'type': 'confirmatory',
        'headline': 'AI Revolution Creates 97 Million New Jobs',
        'statement': 'T02-S02',
        'weight': 1.0,
        'statements': {
            'T02-S02': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T02D': {
        'type': 'disconfirmatory',
        'headline': 'Automation Threatens 47% of Current Employment',
        'statement': 'T02-S02',
        'weight': 1.0,
        'statements': {
            'T02-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T02-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T02E': {
        'type': 'neutral',
        'headline': 'Digital Technology Impact Varies by Context',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T02-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T02-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T02-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T02-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T02-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T03: Economic Policy and Wealth Distribution
    'T03A': {
        'type': 'confirmatory',
        'headline': 'Minimum Wage Hikes Lead to 8% Job Losses',
        'statement': 'T03-S04',
        'weight': 1.0,
        'statements': {
            'T03-S01': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T03-S04': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T03B': {
        'type': 'disconfirmatory',
        'headline': 'Higher Wages Boost Economy Without Job Losses',
        'statement': 'T03-S04',
        'weight': 1.0,
        'statements': {
            'T03-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T03-S04': {'weight': 1.0, 'relationship': 'direct-disconfirm'}
        }
    },
    'T03C': {
        'type': 'confirmatory',
        'headline': 'Progressive Taxation Reduces Inequality by 23%',
        'statement': 'T03-S05',
        'weight': 1.0,
        'statements': {
            'T03-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T03-S05': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T03D': {
        'type': 'disconfirmatory',
        'headline': 'High Taxes Drive Wealth and Investment Abroad',
        'statement': 'T03-S05',
        'weight': 1.0,
        'statements': {
            'T03-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T03-S05': {'weight': 1.0, 'relationship': 'direct-disconfirm'}
        }
    },
    'T03E': {
        'type': 'neutral',
        'headline': 'Income Solutions Require Multiple Approaches',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T03-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T03-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T03-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T03-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T03-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T04: Health and Medical Approaches
    'T04A': {
        'type': 'confirmatory',
        'headline': 'Psychotherapy Shows 68% Recovery Rate',
        'statement': 'T04-S01',
        'weight': 1.0,
        'statements': {
            'T04-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T04-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T04B': {
        'type': 'disconfirmatory',
        'headline': 'Psychiatric Medications Provide Rapid Relief',
        'statement': 'T04-S01',
        'weight': 1.0,
        'statements': {
            'T04-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T04-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T04C': {
        'type': 'confirmatory',
        'headline': 'Prevention Programs Save $3.20 per Dollar',
        'statement': 'T04-S03',
        'weight': 1.0,
        'statements': {
            'T04-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T04-S03': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T04D': {
        'type': 'disconfirmatory',
        'headline': 'Medical Innovation Extends Life 5.2 Years',
        'statement': 'T04-S03',
        'weight': 1.0,
        'statements': {
            'T04-S03': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T04-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T04E': {
        'type': 'neutral',
        'headline': 'Healthcare Combines Prevention and Treatment',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T04-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T04-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T04-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T04-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T04-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T05: Education and Learning Methods
    'T05A': {
        'type': 'confirmatory',
        'headline': 'Online Education Matches Traditional Learning',
        'statement': 'T05-S01',
        'weight': 1.0,
        'statements': {
            'T05-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T05-S03': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T05B': {
        'type': 'disconfirmatory',
        'headline': 'In-Person Education Shows 15% Higher Graduation',
        'statement': 'T05-S01',
        'weight': 1.0,
        'statements': {
            'T05-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T05-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T05C': {
        'type': 'confirmatory',
        'headline': 'Standardized Tests Predict College Success',
        'statement': 'T05-S02',
        'weight': 1.0,
        'statements': {
            'T05-S02': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T05-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T05D': {
        'type': 'disconfirmatory',
        'headline': 'Testing Narrows Curriculum, Increases Stress',
        'statement': 'T05-S02',
        'weight': 1.0,
        'statements': {
            'T05-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T05-S05': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T05E': {
        'type': 'neutral',
        'headline': 'Educational Assessment Requires Multiple Measures',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T05-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T05-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T05-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T05-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T05-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T06: Artificial Intelligence and Ethics
    'T06A': {
        'type': 'confirmatory',
        'headline': 'AI Diagnosis Outperforms Doctors by 94%',
        'statement': 'T06-S01',
        'weight': 1.0,
        'statements': {
            'T06-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T06-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T06B': {
        'type': 'disconfirmatory',
        'headline': 'Fatal AI Errors Expose Algorithmic Bias',
        'statement': 'T06-S01',
        'weight': 1.0,
        'statements': {
            'T06-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T06-S04': {'weight': 0.5, 'relationship': 'indirect-disconfirm'},
            'T06-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T06C': {
        'type': 'confirmatory',
        'headline': 'Facial Recognition Solves 70% More Crimes',
        'statement': 'T06-S02',
        'weight': 1.0,
        'statements': {
            'T06-S02': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T06D': {
        'type': 'disconfirmatory',
        'headline': 'Mass Surveillance Creates Authoritarian Infrastructure',
        'statement': 'T06-S02',
        'weight': 1.0,
        'statements': {
            'T06-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T06-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T06E': {
        'type': 'neutral',
        'headline': 'AI Governance Requires Balancing Innovation',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T06-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T06-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T06-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T06-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T06-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T07: Work-Life Balance and Productivity
    'T07A': {
        'type': 'confirmatory',
        'headline': 'Four-Day Work Week Increases Productivity 40%',
        'statement': 'T07-S01',
        'weight': 1.0,
        'statements': {
            'T07-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T07-S03': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T07B': {
        'type': 'disconfirmatory',
        'headline': 'Five-Day Structure Essential for Business',
        'statement': 'T07-S01',
        'weight': 1.0,
        'statements': {
            'T07-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T07-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T07C': {
        'type': 'confirmatory',
        'headline': 'Remote Teams Outperform Office Workers',
        'statement': 'T07-S02',
        'weight': 1.0,
        'statements': {
            'T07-S02': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T07-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T07D': {
        'type': 'disconfirmatory',
        'headline': 'In-Person Collaboration Drives 65% More Innovation',
        'statement': 'T07-S02',
        'weight': 1.0,
        'statements': {
            'T07-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T07-S04': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T07E': {
        'type': 'neutral',
        'headline': 'Optimal Work Arrangements Vary by Role',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T07-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T07-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T07-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T07-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T07-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T15: Media and Information
    'T15A': {
        'type': 'confirmatory',
        'headline': 'Professional Journalism 73% More Accurate',
        'statement': 'T15-S01',
        'weight': 1.0,
        'statements': {
            'T15-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T15-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T15B': {
        'type': 'disconfirmatory',
        'headline': 'Citizen Journalists Expose Stories Media Ignores',
        'statement': 'T15-S01',
        'weight': 1.0,
        'statements': {
            'T15-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T15-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T15C': {
        'type': 'confirmatory',
        'headline': 'Fact-Checking Reduces False Belief 30%',
        'statement': 'T15-S02',
        'weight': 1.0,
        'statements': {
            'T15-S02': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T15D': {
        'type': 'disconfirmatory',
        'headline': 'Fact-Checkers Show Political Bias',
        'statement': 'T15-S02',
        'weight': 1.0,
        'statements': {
            'T15-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T15-S03': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T15E': {
        'type': 'neutral',
        'headline': 'Information Quality Requires Multiple Sources',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T15-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T15-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T15-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T15-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T15-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T16: Science and Research Funding
    'T16A': {
        'type': 'confirmatory',
        'headline': 'Basic Science Yields $5 Return per Dollar',
        'statement': 'T16-S01',
        'weight': 1.0,
        'statements': {
            'T16-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T16-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T16B': {
        'type': 'disconfirmatory',
        'headline': 'Applied Research Solves Immediate Crises',
        'statement': 'T16-S01',
        'weight': 1.0,
        'statements': {
            'T16-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T16-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T16C': {
        'type': 'confirmatory',
        'headline': 'Animal Research Essential for Medical Breakthroughs',
        'statement': 'T16-S02',
        'weight': 1.0,
        'statements': {
            'T16-S02': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T16D': {
        'type': 'disconfirmatory',
        'headline': 'Advanced Alternatives Make Animal Testing Unnecessary',
        'statement': 'T16-S02',
        'weight': 1.0,
        'statements': {
            'T16-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T16-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T16E': {
        'type': 'neutral',
        'headline': 'Scientific Progress Requires Portfolio Approach',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T16-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T16-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T16-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T16-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T16-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T17: Parenting and Child Development
    'T17A': {
        'type': 'confirmatory',
        'headline': 'Two-Hour Screen Limit Improves Cognition 25%',
        'statement': 'T17-S01',
        'weight': 1.0,
        'statements': {
            'T17-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T17-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T17B': {
        'type': 'disconfirmatory',
        'headline': 'Educational Apps Accelerate Learning',
        'statement': 'T17-S01',
        'weight': 1.0,
        'statements': {
            'T17-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T17-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T17C': {
        'type': 'disconfirmatory',
        'headline': 'Clear Rules Produce Successful Adults',
        'statement': 'T17-S02',
        'weight': 1.0,
        'statements': {
            'T17-S02': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T17-S04': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T17D': {
        'type': 'confirmatory',
        'headline': 'Unstructured Childhood Fosters Creativity',
        'statement': 'T17-S02',
        'weight': 1.0,
        'statements': {
            'T17-S02': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T17-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T17E': {
        'type': 'neutral',
        'headline': 'Effective Parenting Adapts to Individual Needs',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T17-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T17-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T17-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T17-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T17-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T18: Aging and Elder Care
    'T18A': {
        'type': 'confirmatory',
        'headline': 'Forced Retirement Wastes Talent',
        'statement': 'T18-S01',
        'weight': 1.0,
        'statements': {
            'T18-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T18-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T18-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T18B': {
        'type': 'disconfirmatory',
        'headline': 'Mandatory Retirement Opens Opportunities',
        'statement': 'T18-S01',
        'weight': 1.0,
        'statements': {
            'T18-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T18-S05': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T18C': {
        'type': 'confirmatory',
        'headline': 'Aging in Place Improves Health 40%',
        'statement': 'T18-S04',
        'weight': 1.0,
        'statements': {
            'T18-S03': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T18-S04': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T18D': {
        'type': 'disconfirmatory',
        'headline': 'Professional Elder Care Prevents 60% Deaths',
        'statement': 'T18-S04',
        'weight': 1.0,
        'statements': {
            'T18-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T18-S04': {'weight': 1.0, 'relationship': 'direct-disconfirm'}
        }
    },
    'T18E': {
        'type': 'neutral',
        'headline': 'Aging Care Requires Personalized Solutions',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T18-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T18-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T18-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T18-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T18-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    },
    
    # Topic T20: Mental Health and Wellness
    'T20A': {
        'type': 'confirmatory',
        'headline': 'Mindfulness as Effective as CBT',
        'statement': 'T20-S01',
        'weight': 1.0,
        'statements': {
            'T20-S01': {'weight': 1.0, 'relationship': 'direct-confirm'},
            'T20-S05': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T20B': {
        'type': 'disconfirmatory',
        'headline': 'Self-Help Cannot Replace Professional Treatment',
        'statement': 'T20-S01',
        'weight': 1.0,
        'statements': {
            'T20-S01': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T20-S04': {'weight': 0.5, 'relationship': 'indirect-disconfirm'}
        }
    },
    'T20C': {
        'type': 'confirmatory',
        'headline': 'Instagram Use Increases Teen Depression 35%',
        'statement': 'T20-S03',
        'weight': 1.0,
        'statements': {
            'T20-S02': {'weight': 0.5, 'relationship': 'indirect-confirm'},
            'T20-S03': {'weight': 1.0, 'relationship': 'direct-confirm'}
        }
    },
    'T20D': {
        'type': 'disconfirmatory',
        'headline': 'Depression Drives Social Media Use',
        'statement': 'T20-S03',
        'weight': 1.0,
        'statements': {
            'T20-S03': {'weight': 1.0, 'relationship': 'direct-disconfirm'},
            'T20-S05': {'weight': 0.5, 'relationship': 'indirect-confirm'}
        }
    },
    'T20E': {
        'type': 'neutral',
        'headline': 'Mental Health Requires Individualized Approaches',
        'statement': 'multiple',
        'weight': 0.3,
        'statements': {
            'T20-S01': {'weight': 0.3, 'relationship': 'neutral'},
            'T20-S02': {'weight': 0.3, 'relationship': 'neutral'},
            'T20-S03': {'weight': 0.3, 'relationship': 'neutral'},
            'T20-S04': {'weight': 0.3, 'relationship': 'neutral'},
            'T20-S05': {'weight': 0.3, 'relationship': 'neutral'}
        }
    }
}

def get_article_info(article_code: str) -> dict:
    """
    Get article information including linked statement and type.
    Backward compatible function for existing code.
    """
    default_response = {'statement': None, 'type': 'unknown', 'weight': 0}
    
    if article_code not in ARTICLE_STATEMENT_MAP:
        return default_response
    
    article_data = ARTICLE_STATEMENT_MAP[article_code]
    
    return {
        'statement': article_data.get('statement'),
        'type': article_data.get('type', 'unknown'),
        'weight': article_data.get('weight', 0)
    }

def get_linked_statement(article_code: str) -> str:
    """
    Get the primary statement that an article relates to.
    Returns the statement with highest weight.
    """
    info = get_article_info(article_code)
    return info.get('statement')

def get_article_type(article_code: str) -> str:
    """Get the bias type of an article"""
    info = get_article_info(article_code)
    return info.get('type', 'unknown')

def get_article_relationships(article_code: str) -> dict:
    """
    Get full article relationships including all linked statements.
    For advanced analysis with multiple statement relationships.
    """
    if article_code not in ARTICLE_STATEMENT_MAP:
        return {}
    
    return ARTICLE_STATEMENT_MAP[article_code].get('statements', {})

def get_primary_statement_advanced(article_code: str) -> tuple:
    """
    Get primary statement and its weight for advanced analysis.
    Returns: (statement_code, weight, relationship)
    """
    if article_code not in ARTICLE_STATEMENT_MAP:
        return None, 0, 'unknown'
    
    statements = ARTICLE_STATEMENT_MAP[article_code].get('statements', {})
    
    if not statements:
        return None, 0, 'unknown'
    
    # Find statement with highest weight
    max_weight = 0
    primary_statement = None
    primary_relationship = 'unknown'
    
    for stmt_code, rel_info in statements.items():
        if rel_info['weight'] > max_weight:
            max_weight = rel_info['weight']
            primary_statement = stmt_code
            primary_relationship = rel_info.get('relationship', 'unknown')
    
    return primary_statement, max_weight, primary_relationship