def get_hospital_tier(score):

    if score >= 7:
        return "Tier 1 Trauma Center"

    elif score >= 4:
        return "Tier 2 Hospital"

    else:
        return "Tier 3 Clinic"