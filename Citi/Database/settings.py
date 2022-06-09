from collections import OrderedDict

REPORT_TYPE_MAPPING = {'-1': 'Catalyst Watch',
                        '0': 'Earning\'s Review', '1': 'Rating Change',
                       '2': 'Target Price Change', '2.1': 'Target Price Increase', '2.2': 'Target Price Decrease',
                       '3': 'Estimate Change',
                       '4': 'Earning\'s Preview',  # mutually exclusive with Earning's Review
                       '5': 'Initiation',
                       '6': 'ad-hoc'
                       }

REPORT_TYPE_DICT = OrderedDict({'Catalyst Watch': 'Catalyst Watch',
                                'Earnings Review': 'Earnings Review', 'Earnings Preview': 'Earnings Review',
                                'Rating Change': 'Rating Change',
                                'Target Price Change': 'Target Price Change',
                                'Estimate Change': 'Estimate Change',
                                'M&A/Divestiture': 'M&A',
                                'M&A': 'M&A'
})

REPORT_TYPE_GLOBAL_DICT = {'Earnings Review': 'QR', 'Earnings Preview': 'QR',
                           'Rating Change': 'RC',
                           'Target Price Change': 'TPC',
                           'Estimate Change': 'EC',
                           'M&A': 'AH',
                           'ad-hoc': 'AH',
                           'Catalyst Watch': 'CW'}
