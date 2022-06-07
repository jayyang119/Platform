# REPORT_TYPE_MAPPING = {'0': 'Rating Change', '1': 'Target Price Change', '2': 'Estimate Change',
#                        '3': 'Earning\'s Review', '4': 'Earning\'s Preview',
#                        '5': 'Target Price Increase', '6': 'Target Price Decrease'}

from collections import OrderedDict

REPORT_TYPE_MAPPING = {'0': 'Earning\'s Review', '1': 'Rating Change',
                       '2': 'Target Price Change', '2.1': 'Target Price Increase', '2.2': 'Target Price Decrease',
                       '3': 'Estimate Change',
                       '4': 'Earning\'s Preview',  # mutually exclusive with Earning's Review
                       '5': 'Initiation',
                       '6': 'ad-hoc'
                       }

REPORT_TYPE_DICT = OrderedDict({'Earnings': 'Earning\'s Review', 'Earning\'s Review': 'Earning\'s Review',
                                'Rating Downgrade': 'Rating Change', 'Rating Upgrade': 'Rating Change',
                                'Price Target Decrease >= 10%': 'Target Price Decrease', 'Price Target Increase >= 10%': 'Target Price Increase',
                                'EPS Estimate Change': 'Estimate Change',
                                'EPS Estimate Decrease >= 10%': 'Estimate Change', 'EPS Estimate Increase >= 10%': 'Estimate Change',
                                'EPS Estimate Decrease': 'Estimate Change', 'EPS Estimate Increase': 'Estimate Change',
                                'Initiation': 'Initiation'
})

REPORT_TYPE_GLOBAL_DICT = {'Earning\'s Review': 'QR', 'Earnings Review': 'QT', 'Earnings Preview': 'QR',
                           'Rating Change': 'RC',
                           'Target Price Change': 'TPC',
                           'Estimate Change': 'EC',
                           'M&A': 'AH',
                           'ad-hoc': 'AH'}
