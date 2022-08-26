import argparse
from snarkthat_main import SnarkThat

parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest='command')

export = subparser.add_parser('export')
prove = subparser.add_parser('prove')
verify = subparser.add_parser('verify')
plot = subparser.add_parser('plot')
export_alt = subparser.add_parser('xport_alt')
prove_alt = subparser.add_parser('prove_alt')
verify_alt = subparser.add_parser('verify_alt')
plot_alt = subparser.add_parser('plot_alt')
compare = subparser.add_parser('compare')


export.add_argument('--r', type=str)
export_alt.add_argument('--r', type=str)
prove.add_argument('--a', type=int, required=True)
prove.add_argument('--b', type=int, required=True)
prove_alt.add_argument('--a', type=int, required=True)
prove_alt.add_argument('--b', type=int, required=True)
verify.add_argument('--a', type=int, required=True)
verify.add_argument('--b', type=int, required=True)
verify_alt.add_argument('--a', type=int, required=True)
verify_alt.add_argument('--b', type=int, required=True)
plot.add_argument('--p', type=str)
plot_alt.add_argument('--p', type=str)


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
    
#alt logic for comparison:

elif args.command == 'export_alt':
    snarkthat = SnarkThat()
    print('alternative R1CS: \n', snarkthat.alt_r1cs)


elif args.command == 'prove_alt':
    snarkthat = SnarkThat()
    print('alternative R1CS: \n', snarkthat.r1cs, '\n alternative witness, S =', snarkthat.alt_witness(args.a,args.b))
    
elif args.command == 'verify_alt':
    snarkthat = SnarkThat()
    print(snarkthat.alt_verify(args.a,args.b))
    
elif args.command == 'plot_alt':
    snarkthat = SnarkThat()
    snarkthat.alt_plot()

# #Comparison of both main and alternative logic:
# elif args.command == 'compare':
#     snarkthat = SnarkThat()
#     snarkthat.alt_plot()
#     snarkthat.poly_plot()
#     snarkthat.alt_verify(args.a,args.b)
#     snarkthat.verify(args.a, args.b)
