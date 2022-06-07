def rc_filter(headline_list: list, summary_list: list) -> list:
    keywords = ['to outperform', 'to underperform', 'to market perform', 'to strong buy']

    summary_contains_rc = summary_list.apply(lambda x: x.lower()).str.contains('|'.join(keywords))
    headline_contains_rc = headline_list.apply(lambda x: x.lower()).str.contains('|'.join(keywords))
    not_contain_io = ~ (
                summary_list.apply(lambda x: x.lower()).str.contains('initia|reiterat|maintain') |
                summary_list.apply(lambda x: x.lower()).str.contains('initia|maintain')
    )
     # (df.rating_curr != df.rating_prev)
    return (summary_contains_rc | headline_contains_rc) & (not_contain_io)


def er_filter(headline_list, summary_list) -> list:
    keyword_list = ['q1', 'q2', 'q3', 'q4', 'h1', 'h2', 'fy', 'profit alert', 'profit warning', '1q', '2q', '3q', '4q']
    keywords = "|".join(keyword_list)
    preliminary = "|".join(['preliminary'])

    return (summary_list.apply(lambda x: x.lower()).str.contains(keywords, regex=True)) | \
           (headline_list.apply(lambda x: x.lower()).str.contains(keywords, regex=True))


def io_filter(headline_list, summary_list) -> list:
    keywords = ['initiate', 'initiation', 'initiating']
    summary_contains_io = summary_list.apply(lambda x: x.lower()).str.contains('|'.join(keywords))
    headline_contains_io = headline_list.apply(lambda x: x.lower()).str.contains('|'.join(keywords))
    summary_not_contains_maintain = ~summary_list.apply(lambda x: x.lower()).str.contains('maintain|reiterate')
    headline_not_contains_maintain = ~headline_list.apply(lambda x: x.lower()).str.contains('maintain|reiterate')

    return (summary_contains_io | headline_contains_io) & summary_not_contains_maintain & headline_not_contains_maintain


def ec_filter(headline_list, summary_list) -> list:
    headline_keywords = ['estimate', 'change', 'tweaking', 'trimming', 'refreshing', 'adjusting', 'updating',
                         'raising estimate', 'cutting estimate', 'reducing estimate']
    summary_keywords = ['tweaking', 'trimming', 'refreshing', 'adjusting', 'updating', 'raising estimate',
                        'cutting estimate', 'reducing estimate']
    summary_contains_ec = summary_list.apply(lambda x: x.lower()).str.contains('|'.join(summary_keywords))
    headline_contains_ec = headline_list.apply(lambda x: x.lower()).str.contains('|'.join(headline_keywords))

    return headline_contains_ec | summary_contains_ec



if __name__ == '__main__':
    

    pass

