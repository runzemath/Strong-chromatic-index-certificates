"""Check the local flag certificates, offer estimates, and palette recurrence."""

from fractions import Fraction as Q
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
import argparse
import json
import os
import re
import shutil
import subprocess

from verify import verify_sdp
from verify_gain6 import verify_occupancy
from verify_mass import COEFFICIENT, verify_mass
from verify_offer_extensions import verify_extensions
from verify_palette import Schedule, verify_palette
from verify_upstream import verify_upstream


ROOT=Path(__file__).resolve().parent
M=10**8
SCALE=10**12


def verify_manifest():
    seen = set()
    for line in (ROOT / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = Path(name)
        assert not path.is_absolute() and '..' not in path.parts
        assert name not in seen and re.fullmatch(r'[0-9a-f]{64}', digest)
        assert sha256((ROOT / path).read_bytes()).hexdigest() == digest, (
            f'SHA-256 mismatch for {name}'
        )
        seen.add(name)
    assert {'README.md', 'parameters.json', 'verify_all.py', 'flags.txt',
            'flags.sdpa', 'schedule.txt', 'verify_palette.py', 'palette_bounds.txt',
            'verify_palette.cpp'} <= seen
    print(f'Checksums verified for {len(seen)} files.', flush=True)


def verify_schedule(path, certified, parameters):
    with path.open() as source:
        n,q,stop,tolerance,claim,count,nrows,method=map(int,source.readline().split())
        density=[tuple(map(int,source.readline().split())) for _ in range(nrows)]
        cuts=list(map(int,source.readline().split()))
        bonus=[tuple(map(int,line.split())) for line in source if line.strip()]
    assert len(cuts)==n and len(bonus)==count
    assert all(x<y for x,y in zip(cuts,cuts[1:])) and cuts[-1]<M
    tau=Q(cuts[0],M)
    assert tau==Q(parameters['threshold']) and Q(claim,SCALE)==tau
    assert Q(parameters['bound'])==2*tau
    assert (n,q,stop,tolerance,count,nrows,method)==(176,500,2200000000,2100,18,5,10)
    assert method==parameters['method']
    assert [(Q(b,M),Q(g,M)) for b,g in density]==[(b,g) for _,g,b in certified]
    assert all(theta<=tau for theta,_,_ in certified)
    assert Q('1.8548')<60*COEFFICIENT<Q('1.85481')
    cases=json.loads((ROOT/'gain6_witness.json').read_text())['cases'][:-1]
    expected=[(25000000,284000),(25500000,311000)]
    expected.extend((int(Q(c['lo'])*M),int(Q(c['bonus'])*M)) for c in cases)
    assert bonus==expected
    return Schedule(q, stop, claim, tuple(cuts), tuple(density), tuple(bonus))


def verify_palette_cpp(path, parameters, compiler):
    with TemporaryDirectory() as tmp:
        binary=Path(tmp)/('verify_palette.exe' if os.name == 'nt' else 'verify_palette')
        subprocess.run([compiler,'-O3','-std=c++20','-Wall','-Wextra','-fopenmp',
                        str(ROOT/'verify_palette.cpp'),'-o',str(binary)],check=True)
        env=dict(os.environ)
        env.setdefault('OMP_NUM_THREADS','4')
        with path.open() as source:
            result=subprocess.run([str(binary)],stdin=source,stdout=subprocess.PIPE,
                                  text=True,check=True,env=env)
    match=re.fullmatch(r'Palette cap (\d+) / (\d+), phases (\d+), boxes (\d+)\s*',result.stdout)
    assert match is not None
    cap,scale,phases,boxes=map(int,match.groups())
    assert (cap,scale,phases,boxes)==(parameters['cap'],SCALE,parameters['phases'],parameters['boxes'])
    assert Q(cap,scale)<Q(parameters['threshold'])
    print(result.stdout,end='',flush=True)


def main():
    if not __debug__:
        raise SystemExit('Run without Python optimization.')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--appendix',choices=('A','B','C'),
                        help='replay only the checks for the specified appendix')
    parser.add_argument('--jobs', type=int, default=min(8, os.cpu_count() or 1),
                        help='number of Python processes for Appendix C (default: at most 8)')
    parser.add_argument('--cpp', action='store_true',
                        help='replay the original C++ search for Appendix C')
    args=parser.parse_args()
    if not 1 <= args.jobs <= 61:
        parser.error('--jobs must be between 1 and 61')
    section=args.appendix
    compiler=None
    if args.cpp and section in (None,'C'):
        compiler=shutil.which('g++')
        if compiler is None:
            parser.error('--cpp requires g++. Run without --cpp to verify with Python.')
    verify_manifest()
    parameters=json.loads((ROOT/'parameters.json').read_text())
    if section in (None,'A'):
        verify_upstream()
        verify_sdp()
        certified=[verify_mass(ROOT/name) for name in parameters['local_certificates']]
    if section in (None,'B'):
        verify_occupancy()
        verify_extensions(sharper=True)
    if section in (None,'C'):
        if section=='C':
            # In isolation, validate the schedule against the witness headers.
            headers=[json.loads((ROOT/name).read_text())
                     for name in parameters['local_certificates']]
            certified=[(Q(h['threshold']),Q(h['slope']),Q(h['intercept']))
                       for h in headers]
        schedule_path=ROOT/'schedule.txt'
        schedule=verify_schedule(schedule_path,certified,parameters)
        if compiler is None:
            verify_palette(schedule,parameters,args.jobs)
        else:
            verify_palette_cpp(schedule_path,parameters,compiler)
    if section is None:
        print('All certificate checks passed.',flush=True)
    else:
        print(f'Appendix {section} checks passed.',flush=True)


if __name__=='__main__':
    main()
