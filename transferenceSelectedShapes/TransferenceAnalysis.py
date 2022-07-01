import pickle
COMPUTER_NAME = '/home/renata'


def transference_dict(SELECT_ROBOT,encode):
    """
    Construct dict with basic info about robots that were transfered
    """

    RUN_DIR = "CPPN_Inovation"
    
    try:
        with open('{2}/locomotion_principles/transferenceSelectedShapes/TransferedDict_4_{0}_{1}'.format(RUN_DIR,SELECT_ROBOT,COMPUTER_NAME ), 'rb') as handle:
            if encode == "ASCII":
                plot_df = pickle.load(handle)
            else:
                plot_df = pickle.load(handle,encoding=encode)
    
    except:
        print('Problem with Transference Dict')

    return plot_df