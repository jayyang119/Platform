from collections import OrderedDict

REPORT_TYPE_MAPPING = {'-1': 'ad-hoc',
                       '0': 'Earning\'s Review',
                       '1': 'Rating Change',
                       '2': 'Target Price Change',
                       '3': 'Estimate Change',
                       '4': 'Initiation'}

REPORT_TYPE_DICT = OrderedDict({'Catalyst Watch': 'Catalyst Watch',
                                'Earnings Review': 'Earnings Review', 'Earnings Preview': 'Earnings Review',
                                'Rating Change': 'Rating Change',
                                'Target Price Change': 'Target Price Change',
                                'Estimate Change': 'Estimate Change',
                                'M&A/Divestiture': 'M&A',
                                'M&A': 'M&A'
})

REPORT_TYPE_GLOBAL_DICT = {'Earning\'s Review': 'ER',
                           'Rating Change': 'RC',
                           'Target Price Change': 'TPC',
                           'Estimate Change': 'EC',
                           'ad-hoc': 'AH',
                           'Initiation': 'IO'}
