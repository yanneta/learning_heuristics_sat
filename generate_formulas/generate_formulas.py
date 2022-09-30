from __future__ import print_function

import argparse
import os
import pdb
import subprocess


def create_sat_problem(filename, call_list):
    while True:
        subprocess.call(call_list)
        try:
            subprocess.check_call(['minisat', 'tmp.cnf'], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:
                os.rename('tmp.cnf', filename)
                return
            os.remove('tmp.cnf')

def generate_k_color(N, n, p, k, i0, path):
    """ 
    n number of nodes
    p probability of edge
    k number of colors
    i0 first id of the formula
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)
    for i in range(N):
        filename = 'kcolor_n{}_p{}_k{}_{}.cnf'.format(n, p, k, i0+i)
        call_list = ['cnfgen', '-q', '-o', 'tmp.cnf', 'kcolor', str(k), 'gnp', str(n), str(p)]
        create_sat_problem(filename, call_list)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expr', type=str, help='experiment name')
    args = parser.parse_args()

    N=2000
    if args.expr == "kcolor/3-5-0.5/":
        generate_k_color(N, 5, 0.5, 3, 0, "data/" + args.expr)
    if args.expr == "kcolor/3-10-0.5/":
        generate_k_color(N, 10, 0.5, 3, 0, "data/" + args.expr)
    if args.expr == "kcolor/4-15-0.5/":
        generate_k_color(N, 15, 0.5, 4, 0, "data/" + args.expr)
    if args.expr == "kcolor/5-20-0.5/":
        generate_k_color(N, 20, 0.5, 5, 0, "data/" + args.expr)

if __name__ == '__main__':
    main()
