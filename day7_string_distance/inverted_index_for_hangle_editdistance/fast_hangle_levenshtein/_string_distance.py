from ._hangle import decompose
from ._hangle import character_is_korean

def levenshtein(s1, s2, cost=None):
    # based on Wikipedia/Levenshtein_distance#Python
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)
    
    if not cost:
        cost = {}
    
    def get_cost(c1, c2, cost):
        return 0 if (c1 == c2) else cost.get((c1, c2), 1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_cost(c1, c2, cost)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def jamo_levenshtein(s1, s2):
    if len(s1) < len(s2):
        return jamo_levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)
    
    def get_jamo_cost(c1, c2):
        c1_is_korean, c2_is_korean = character_is_korean(c1), character_is_korean(c2)
        if c1_is_korean and c2_is_korean:
            return 0 if (c1 == c2) else levenshtein(decompose(c1), decompose(c2))/3
        elif not (c1_is_korean) and not (c2_is_korean):
            return 0 if (c1 == c2) else 1
        else:
            return 1

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + get_jamo_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]