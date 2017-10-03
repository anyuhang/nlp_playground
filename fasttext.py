from sklearn.model_selection import train_test_split

def removePunctations(text):
  return ''.join(t for t in text if t.isalnum() or t == ' ')

def get_pa():
    questions = []
    pa_list = []
    for question in open('questions_v3.tsv'):
        question, pa = question.split('\t')
        pa_name = d.keys()[d.values().index(int(pa.strip("\n")))]
        pa_list.append(pa.strip("\n"))
        questions.append('__label__'+ pa_name + ' ' + removePunctations(question).strip(" "))
    return questions, pa_list

sen, pa = get_pa()
X_train, X_test= train_test_split(sen, test_size=0.1, stratify=pa, random_state=11)

with open('questions_train.tsv', 'w') as f:
    for i in X_train:
        f.write(i + '\n')

with open('questions_test.tsv', 'w') as f:
    for i in X_test:
        f.write(i + '\n')

./fasttext supervised -input questions_train.tsv -output model_qa

./fasttext test model_qa.bin questions_test.tsv

d = {}
d['Real_Estate'] = 112
d['Bankruptcy_Debt'] = 2
d['Trademark_Application'] = 102
d['Antitrust_Trade_Law'] = 12
d['Federal_Regulation'] = 92
d['Life_Sciences_Biotechnology'] = 32
d['Child_Custody'] = 82
d['Venture_Capital'] = 42
d['Guardianship'] = 72
d['Mesothelioma_Asbestos'] = 132
d['Education'] = 62
d['Equipment_Finance_and_Leasing'] = 22
d['Motorcycle_Accident'] = 122
d['Insurance_Fraud'] = 52
d['Slip_and_Fall_Accident'] = 123
d['Civil_Rights'] = 43
d['Aviation'] = 13
d['Employment_Labor'] = 63
d['Chapter_7_Bankruptcy'] = 3
d['Financial_Markets_and_Services'] = 23
d['Child_Support'] = 83
d['State,_Local_And_Municipal_Law'] = 93
d['Mergers_Acquisitions'] = 33
d['Commercial'] = 113
d['Animal_Dog_Bites'] = 133
d['Securities_Investment_Fraud'] = 53
d['Power_Of_Attorney'] = 73
d['International_Law'] = 103
d['Personal_Injury'] = 104
d['Tax_Fraud_Tax_Evasion'] = 54
d['Military_Law'] = 94
d['Construction_Development'] = 114
d['Franchising'] = 24
d['Discrimination'] = 64
d['Oil_Gas'] = 34
d['Banking'] = 14
d['Expungement'] = 134
d['Probate'] = 74
d['Constitutional'] = 44
d['Chapter_11_Bankruptcy'] = 4
d['Trucking_Accident'] = 124
d['Divorce_Separation'] = 84
d['Asylum'] = 137
d['Arbitration'] = 127
d['Landlord_Tenant'] = 117
d['Defective_and_Dangerous_Products'] = 107
d['Copyright_Infringement'] = 97
d['Uncontested_Divorce'] = 87
d['Ethics_Professional_Responsibility'] = 77
d['Workers_Compensation'] = 67
d['Federal_Crime'] = 57
d['Consumer_Protection'] = 47
d['Public_Finance_Tax_Exempt_Finance'] = 37
d['Health_Care'] = 27
d['Corporate_Incorporation'] = 17
d['Credit_Repair'] = 7
d['Life_Insurance'] = 138
d['Lemon_Law'] = 48
d['Juvenile'] = 58
d['Wrongful_Termination'] = 68
d['Debt_Lending_Agreements'] = 18
d['Family'] = 78
d['General_Practice'] = 88
d['Securities_Offerings'] = 38
d['Copyright_Application'] = 98
d['Libel_Slander'] = 108
d['Debt_Settlement'] = 8
d['Class_Action'] = 128
d['Insurance'] = 28
d['Residential'] = 119
d['Government'] = 89
d['Tax'] = 39
d['Patent_Infringement'] = 99
d['Computer_Fraud'] = 49
d['Mediation'] = 129
d['Speeding_Traffic_Ticket'] = 59
d['Business'] = 9
d['Employee_Benefits'] = 19
d['Internet'] = 29
d['Medicaid_Medicare'] = 139
d['Environmental_and_Natural_Resources'] = 69
d['Alimony'] = 79
d['Medical_Malpractice'] = 109
d['Spinal_Cord_Injury'] = 135
d['Immigration'] = 95
d['Domestic_Violence'] = 85
d['Lawsuits_Disputes'] = 125
d['Gaming'] = 25
d['Communications_Media'] = 15
d['Chapter_13_Bankruptcy'] = 5
d['Trusts'] = 75
d['Sexual_Harassment'] = 65
d['Foreclosure'] = 115
d['Native_Peoples_Law'] = 45
d['Criminal_Defense'] = 55
d['Birth_Injury'] = 105
d['Partnership'] = 35
d['Marriage_Prenuptials'] = 86
d['Appeals'] = 126
d['Debt_Collection'] = 6
d['Project_Finance'] = 36
d['Sex_Crime'] = 136
d['Brain_Injury'] = 106
d['Intellectual_Property'] = 96
d['Privacy'] = 46
d['Wills_Living_Wills'] = 76
d['Contracts_Agreements'] = 16
d['Government_Contracts'] = 26
d['Social_Security'] = 66
d['Land_Use_Zoning'] = 116
d['DUI_DWI'] = 56
d['Nursing_Home_Abuse_and_Neglect'] = 110
d['Advertising'] = 120
d['Litigation'] = 130
d['Credit_Card_Fraud'] = 50
d['Violent_Crime'] = 60
d['Energy_Utilities'] = 20
d['Health_Insurance'] = 140
d['Estate_Planning'] = 70
d['Adoption'] = 80
d['Administrative_Law'] = 90
d['Telecommunications'] = 40
d['Licensing'] = 30
d['Patent_Application'] = 100
d['Admiralty_Maritime'] = 10
d['Transportation'] = 41
d['Limited_Liability_Company_(LLC)'] = 31
d['Car_Accidents'] = 121
d['Unknown'] = 1
d['Wrongful_Death'] = 111
d['Trademark_Infringement'] = 101
d['Agriculture'] = 11
d['Election_Campaigns_Political_Law'] = 91
d['Child_Abuse'] = 81
d['Elder_Law'] = 71
d['White_Collar_Crime'] = 61
d['Entertainment'] = 21
d['Animal_Law'] = 131
d['Identity_Theft'] = 51
