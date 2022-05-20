#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright 2020 the HERA Project
# Licensed under the MIT License

"""Command-line script for querying the HERA CM database for bad antennas"""

import argparse
from hera_mc import cm_active


def query_ex_ants(JD, good_statuses):
    '''Query the HERA CM database for antennas considered bad on a certain date.

    Arguments
        JD: string, int, or float Julian Date on which to queury the database
        good_statuses: string of comma-separated statuses considered acceptable. Antennas
            with any other status will be returned. Current possibilities include:
                * 'dish_maintenance',
                * 'dish_ok',
                * 'RF_maintenance',
                * 'RF_ok',
                * 'digital_maintenance',
                * 'digital_ok',
                * 'calibration_maintenance',
                * 'calibration_ok',
                * 'calibration_triage'

    Returns:
        ex_ants: string of space-separated antennas considered bad
    '''
    # Load database
    h = cm_active.get_active(at_date=JD, float_format='jd')

    # Check that input statuses are sensible
    good_statuses = [status.strip() for status in good_statuses.split(',')]
    assert len(good_statuses) > 0, 'There must be at least one input good status.'
    valid_statuses = list(h.apriori.values())[0].valid_statuses() + list(h.apriori.values())[0].old_statuses()
    for status in good_statuses:
        assert status in valid_statuses, 'Invalid Status: {}'.format(status)

    # Pick out antnenna names with bad statuses
    exants = []
    for ant in h.apriori:
        if h.apriori[ant].status not in good_statuses:
            exants.append(int(ant[2:].split(':')[0]))  # Assumes the format HH0:A or HA330:A

    # Return sorted exants
    exants = ' '.join([str(ant) for ant in sorted(exants)])
    return exants


# Parse arguments
a = argparse.ArgumentParser(description='Command line function for printing out space-separated "bad" antennas on a given JD.')
a.add_argument("JD", type=float, help="Julian data on which to query the database.")
a.add_argument("good_statuses", type=str, help="Comma-separated list of acceptable anntenna statuses. Antennas "
                                               "with any other status will be printed out to the command line.")
args = a.parse_args()

# Print ex_ants
print(query_ex_ants(args.JD, args.good_statuses))
