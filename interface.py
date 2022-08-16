import argparse
from snarkthat import SnarkThat

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')

export = subparser.add_parser('export')
prove = subparser.add_parser('prove')
verify = subparser.add_parser('verify')

export.add_argument('', type=str)
prove.add_argument('--a', type=int, required=True)
prove.add_argument('--b', type=int, required=True)
verify.add_argument('--a', type=int, required=True)
verify.add_argument('--b', type=int, required=True)


args = parser.parse_args()
if args.command == 'export':
    snarkthat = SnarkThat()
    print('R1CS: \n', snarkthat.r1cs)


elif args.command == 'prove':
    snarkthat = SnarkThat()
    print('R1CS: \n', snarkthat.r1cs, '\n S =', snarkthat.arithmetic_circuit(args.a,args.b))
    
#   'for new member', args.firstname, args.lastname,
#   'with email:', args.email,
#   'and password:', args.password)
elif args.command == 'verify':
    snarkthat = SnarkThat()
    print(snarkthat.verify(args.a,args.b))
