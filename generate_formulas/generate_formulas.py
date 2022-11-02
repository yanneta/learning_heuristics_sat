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

def generate_k_color(N, n, p, k, i0):
    """ 
    n number of nodes
    p probability of edge
    k number of colors
    i0 first id of the formula
    """
    for i in range(N):
        filename = 'kcolor_n{}_p{}_k{}_{}.cnf'.format(n, p, k, i0+i)
        call_list = ['cnfgen', '-q', '-o', 'tmp.cnf', 'kcolor', str(k), 'gnp', str(n), str(p)]
        create_sat_problem(filename, call_list)

def generate_randksat(N, k, n, m, i0):
    """ 
    n number of variables
    m number of clauses
    k number literals per clause
    i0 first id of the formula
    """
    for i in range(N):
        filename = 'ksat_k{}_n{}_m{}_{}.cnf'.format(k, n, m, i0+i)
        call_list = ['cnfgen', '-q', '-o', 'tmp.cnf', 'randkcnf', str(k), str(n), str(m)]
        create_sat_problem(filename, call_list)

def generate_kclique(N, n, p, k, i0):
    """
    n number of nodes
    p probability of an edge
    k size of the clique
    """
    for i in range(N):
        filename = 'kclique_k{}_n{}_p{}_{}.cnf'.format(k, n, p, i0+i)
        call_list = ['cnfgen', '-q', '-o', 'tmp.cnf', 'kclique', str(k), 'gnp', str(n), str(p)]
        create_sat_problem(filename, call_list)

#cnfgen -q -o tmp.cnf domset 4 gnp 12 0.2

def generate_domeset(N, n, p, k, i0):
    """
    n number of nodes
    p probability of an edge
    k size of the dominating set
    """
    for i in range(N):
        filename = 'domset_k{}_n{}_p{}_{}.cnf'.format(k, n, p, i0+i)
        call_list = ['cnfgen', '-q', '-o', 'tmp.cnf', 'domset', str(k), 'gnp', str(n), str(p)]
        create_sat_problem(filename, call_list)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('expr', type=str, help='experiment name')
    args = parser.parse_args()

    path = "data/" + args.expr
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
    os.chdir(path)

    N=2000

    if args.expr == "domset/3-9-0.2/"
        generate_domeset(N, 9, 0.2, 3, 0)
    
     if args.expr == "domset/4-12-0.2/"
        generate_domeset(N, 12, 0.2, 4, 0)
    
    if args.expr == "kclique/3-5-0.2/":
        generate_kclique(N, 5, 0.2, 3, 0) 

    if args.expr == "kclique/3-20-0.05/":
        generate_kclique(N, 20, 0.05, 3, 0)
    #python code/kclique.py data/kclique/5-0.2/ 2000 5 0.2 3 0 &
    #python code/kclique.py data/kclique/10-0.1/ 2000 10 0.1 3 0 &
    #python code/kclique.py data/kclique/15-0.066/ 2000 15 0.066 3 0 &
    #python code/kclique.py data/kclique/20-0.05/ 2000 20 0.05 3 0 &
    

    # random k-SAT experiments
    if args.expr == "rand3sat/5-21/":
        generate_randksat(N, 3, 5, 21, 0)
    if args.expr == "rand3sat/10-43/":
        generate_randksat(N, 3, 10, 34, 0)
    if args.expr == "rand3sat/25-106/":
        generate_randksat(N, 3, 25, 106, 0)
    if args.expr == "rand3sat":
        generate_randksat(N, 3, 50, 213, 0)
        generate_randksat(N, 3, 75, 320, 0)
        generate_randksat(N, 3, 100, 426, 0)
        generate_randksat(N, 3, 150, 639, 0)
        generate_randksat(N, 3, 200, 852, 0)

    # kcolor experiments
    if args.expr == "kcolor/3-5-0.5/":
        generate_k_color(N, 5, 0.5, 3, 0)
    if args.expr == "kcolor/3-10-0.5/":
        generate_k_color(N, 10, 0.5, 3, 0)
    if args.expr == "kcolor/4-15-0.5/":
        generate_k_color(N, 15, 0.5, 4, 0)
    if args.expr == "kcolor/5-20-0.5/":
        generate_k_color(N, 20, 0.5, 5, 0)

if __name__ == '__main__':
    main()
