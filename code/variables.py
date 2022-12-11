import os

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def is_cue(weight_matrix, cue):
    '''Return True if cue is in weight matrix, return False if not.'''
    if cue in weight_matrix.index:
        return True
    else:
        return False

def is_outcome(weight_matrix, outcome):
    '''Return true if outcome is in weight matrix, return False if not.'''
    if outcome in weight_matrix.columns:
        return True
    else:
        return False

def sum_of_column(weight_matrix, columname):
    """Return the sum of a column."""
    weights = weight_matrix[columname].tolist()
    summe = sum(np.absolute(weights))
    return summe

def sum_of_row(weight_matrix, rowname):
    """Return the sum of a row."""
    weights = weight_matrix.loc[rowname, :].values.tolist()
    summe = sum(np.absolute(weights))
    return summe

def get_all_cues(weight_matrix, outcome, per_domain = False):
    """Get the all cues in a weight matrix or all cues in a weight matrix per domain.

    Input:
    -----
    event_files - list
         A list of event files.
    outcome - str
        The word that the predicting cues should be retrieved for.
    per_domain - bool
        Whether to sort the cues per domain.

    Output:
    -------
    cues - list of str
        A list of the predicting cues for the given outcome taken from the event files.
    """

    all_cues = weight_matrix.index
    if per_domain:
        context = [cue for cue in all_cues if cue.startswith('c.')]
        segments = [cue for cue in all_cues if cue.startswith('s.')]
        syllables = [cue for cue in all_cues if cue.startswith('y.')]
        domain_cues = {'Segment cues: ': segments, 'Syllable cues: ': syllables, 'Context cues: ': context}
        return domain_cues

    else:
        return all_cues

def get_all_predicting_cues(event_files, outcome, per_domain = False, no_context = False):
    """Get all predicting cues, or all predicting cues per domain (context, syllables, segment)
    of an outcome from event files. GETS ALL CUES FORM ALL EVENTS, NOT JUST ONE. 

    Input:
    -----
    event_files - list
         A list of event files.
    outcome - strvsc
        The word that the predicting cues should be retrieved for.
    per_domain - bool
        Whether to sort the predicting cues per domain.
    no_context - bool
        Whether to only return the segments and syllable cues. 

    Output:
    -------
    predicting_cues - list
        A list of the predicting cues for the given outcome taken from the event files.
    domain_cues - dict
        A dict of the predicting cues for each domain for the given outcome, taken from the event files.
    """
    
    # Only look for the outcome once to find an return only syllable and segment cues.
    if no_context: 
        predicting_cues = []
        for file in event_files:
            outcomes = file['outcomes'].tolist()
            for index, word in enumerate(outcomes):
                if word == outcome:
                    cues = file.at[index, 'cues']
                    cues = cues.split('_')
                    for cue in cues:
                        if cue.startswith('y.') or cue.startswith('s.'):
                            predicting_cues.append(cue)
                    return list(set(predicting_cues))
                
    # Look for all context cues for the outcome. 
    else:
        predicting_cues = []
        for file in event_files:
            outcomes = file['outcomes'].tolist()
            for index, word in enumerate(outcomes):
                if word == outcome:
                    cues = file.at[index, 'cues']
                    cues = cues.split('_')
                    for cue in cues:
                        if cue not in predicting_cues:
                            predicting_cues.extend(cues)

    if per_domain:
        context = [cue for cue in predicting_cues if cue.startswith('c.')]
        segments = [cue for cue in predicting_cues if cue.startswith('s.')]
        syllables = [cue for cue in predicting_cues if cue.startswith('y.')]
        domain_cues = {'Segment cues: ': segments, 'Syllable cues: ': syllables, 'Context cues: ': context}

        return domain_cues

    else:
        return predicting_cues
    
def sum_domain_cues(weight_matrix, cues, outcome):
    """ Given a dict of cues, for each domain, sum the weights of the cues to a given outcome and return them
    in a dict.
    """
    context = [weight_matrix.at[cue, outcome] for cue in cues['Context cues: '] if cue.startswith('c.')]
    prior_context = sum(context)

    segments = [weight_matrix.at[cue, outcome] for cue in cues['Segment cues: '] if cue.startswith('s.')]
    prior_segments = sum(segments)

    syllables = [weight_matrix.at[cue, outcome] for cue in cues['Syllable cues: '] if cue.startswith('y.')]
    prior_syllables = sum(syllables)

    all_prior = {'Segment': prior_segments, 'Syllable': prior_syllables, 'Context': prior_context}

    return all_prior

def get_prior(weight_matrix, word_outcome, domain_specific = False):
    ''' Calculate the prior measures of an outcome.
    Takes the sum of all of the weights in the outcome vector, or the ones ones for each domain
    in the outcome vector and sums them.

    Input:
    -----
    weight_matrix - pandas.DataFrame
        A weight matrix from a trained NDL model that has a column containing the sums of the cue vectors
        called 'cue_sums', and a row containing the sums of the outcome vectors called 'outcome_sums'.
    word_outcome - str
        A word that is contained in the outcomes of the weight_matrix.
    domain_specific - bool
        Whether to calculate the prior per domain or not.

    Output:
    ------
    prior - float
        The prior of the input word.
    all_prior - dict
        A dictionary containing the prior for each domain for the given outcome.
    '''

    # Check if outcome is in weight_matrix.
    if is_outcome(weight_matrix = weight_matrix, outcome = word_outcome):
        outcome = word_outcome
    else:
        return 'The model was not trained on the word you requested.'

    # Get the weight of each cue outcome pair for each domain.
    if domain_specific:
        all_prior = {} # Do I need this?
        cues = get_all_cues(weight_matrix = weight_matrix, outcome = outcome, per_domain = True)
        all_prior = sum_domain_cues(weight_matrix = weight_matrix, cues = cues, outcome = outcome)

        return all_prior

    else:
        prior = sum_of_column(weight_matrix = weight_matrix, columname = outcome)
    return prior


def activation(word_outcome, event_files, weight_matrix, c1, c2 = None, domain_specific = False):
    """Returns activation for a specific learning event.
    Input:
    -----
    weight_matrix - pandas.DataFrame
        A weight matrix from a trained NDL model that has a column containing the sums of the cue vectors
        called 'cue_sums', and a row containing the sums of the outcome vectors called 'outcome_sums'.
    word_outcome - str
        An outcome from the weight matrix.
    c1 - str
        A cue-tagged string 'c.it'
    c2 - str
        An optional second cue-tagged string
    domain_specific - bool
        Whether the activation is retrieved per domain or from all domains together. Default is None.
    event_files - list
         A list of event files. Default is None.

    Output:
    -------
    activation - float
        The activation for a given outcome.
    all_activation - dict
        A dict of the activations for each domain.
    """
    
    # Check if the outcome is the weight_matrix.
    if is_outcome(weight_matrix = weight_matrix, outcome = word_outcome):
        outcome = word_outcome
    else:
        return 'The word ' + word_outcome + ' is not in the outcomes of the weight matrix.'
    
    # Get the syllable and segments cues for the outcome. 
    cues = get_all_predicting_cues(event_files = event_files, outcome = outcome, per_domain = False, no_context = True)
    
    if domain_specific: 
        segments = [cue for cue in cues if cue.startswith('s.')]
        syllables = [cue for cue in cues if cue.startswith('y.')]
        if c2: 
            context = [c1, c2]
        else: 
            context = c1
        
        domain_cues = {'Segment cues: ': segments, 'Syllable cues: ': syllables, 'Context cues: ': context}

        all_activation = sum_domain_cues(weight_matrix = weight_matrix, cues = domain_cues, outcome = outcome)

        return all_activation
        
    else:
        all_weights = [] 
        cues.append(c1) # A pretty ugly solution for a problem I was having.
        if c2: 
            cues.append(c2)
        
        for cue in cues:
            weight = weight_matrix.at[cue, outcome]
            all_weights.append(weight)   
        activation = sum(all_weights)
        
        return activation

def get_activation(weight_matrix,  event_files, word_outcome, cues = None, domain_specific = False):
    """Calculates activation for an outcome.
    Given a list of cues that, in the event files that the model was trained with, predict the outcome, sum the weights
    of those cues to the given outcome. Either per domain, or all together. USES THE FUNCTION THAT GETS ALL CUES OVER ALL
    LEARNING EVENTS.

    Input:
    -----
    weight_matrix - pandas.DataFrame
        A weight matrix from a trained NDL model that has a column containing the sums of the cue vectors
        called 'cue_sums', and a row containing the sums of the outcome vectors called 'outcome_sums'.
    word_outcome - str
        An outcome from the weight matrix.
    domain_specific - bool
        Whether the activation is retrieved per domain or from all domains together. Default is None.
    event_files - list
         A list of event files. Default is None.

    Output:
    -------
    activation - float
        The activation for a given cue or outcome.
    all_activation - dict
        A dict of the activations for each domain.
    """

    # Check if the outcome is the weight_matrix.
    if is_outcome(weight_matrix = weight_matrix, outcome = word_outcome):
        outcome = word_outcome
    else:
        return 'The word ' + word_outcome + ' is not in the outcomes of the weight matrix.'

    # Get domain specific activation
    if domain_specific:
        if not cues:
            cues = get_predicting_cues(event_files = event_files, outcome = outcome, per_domain = True)

        all_activation = sum_domain_cues(weight_matrix = weight_matrix, cues = cues, outcome = outcome)
        return all_activation

    # Get general activation
    else:
        all_weights = []

        if not cues:
            cues = get_predicting_cues(event_files = event_files, outcome = outcome, per_domain = False)

        for cue in cues:
            weight = weight_matrix.at[cue, outcome]
            all_weights.append(weight)

        activation = sum(all_weights)
        return activation
    
    
def get_activation_diversity(weight_matrix, event_files, input_word):
    """ Calculate activation diversity for an input word (cue).
    For a given cue, get the activations of the outcomes the cue is connected to and sum them.

    Input:
    -----
    weight_matrix - pandas.DataFrame
        A weight matrix from a trained NDL model that has a column containing the sums of the cue vectors
        called 'cue_sums', and a row containing the sums of the outcome vectors called 'outcome_sums'.
    event_files - list
         A list of event files.
    input_word - str
        A cue from the weight matrix.

    Output:
    ------
    act_div - float
        Activation diversity for the given cue.
    """

    # Check if the cue is in the weight matrix.
    if is_cue(weight_matrix = weight_matrix, cue = input_word):
        cue = input_word
    else:
        return 'The word ' + cue + ' is not in the cues of the weight matrix.'

    # Go through cue strings.
    all_activations = []
    for file in event_files:
        cues = weight_matrix.index
        for index, word in enumerate(cues):
            # If the cue is in there, get the corresponding outcome and its activation.
            if cue in word:
                outcome = file.at[index, 'outcomes']
                cues = get_predicting_cues(event_files = event_files, outcome = outcome, per_domain = False)
                activation = get_activation(cues = cues, domain_specific = False, event_files = event_files,
                                            weight_matrix = weight_matrix, word_outcome = outcome)
                all_activations.append(np.absolute(activation))

    act_div = sum(all_activations)
    return act_div
