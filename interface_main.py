import argparse
from snarkthat_main import SnarkThat
import test_snarkthat

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')

export = subparser.add_parser('export')
prove = subparser.add_parser('prove')
verify = subparser.add_parser('verify')
plot = subparser.add_parser('plot')

export.add_argument('--r', type=str)
prove.add_argument('--a', type=int, required=True)
prove.add_argument('--b', type=int, required=True)
verify.add_argument('--a', type=int, required=True)
verify.add_argument('--b', type=int, required=True)
plot.add_argument('--p', type=str)


args = parser.parse_args()
if args.command == 'export':
    snarkthat = SnarkThat()
    print('R1CS: \n', snarkthat.r1cs)


elif args.command == 'prove':
    snarkthat = SnarkThat()
    print('R1CS: \n', snarkthat.r1cs, '\n S =', snarkthat.arithmetic_circuit(args.a,args.b))
    
elif args.command == 'verify':
    snarkthat = SnarkThat()
    print(snarkthat.verify(args.a,args.b))
    
elif args.command == 'plot':
    snarkthat = SnarkThat()
    snarkthat.plot_poly()
