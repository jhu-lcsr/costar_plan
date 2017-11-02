from __future__ import print_function


def main():
    # fix a csv that was accidentally saved "flat"
    files = os.listdir('.')
    for filename in files:
        if filename[-3:] == 'csv':
            toks = filename.split('.')
            toks[-2] += "_corrected"
            fixed_filename = ''
            for i, t in enumerate(toks):
                fixed_filename += t
                if i < len(toks) - 1:
                    fixed_filename += '.'
            print(filename,' ---> ',fixed_filename)
            with f as open(filename, 'r'):
                with o as open(filename, 'w'):
                    line = f.readline()
                    if line[-1] == '\n':
                        line = line[:-1]
                    toks = line.split('.')
                    newline = ''
                    for i, tok in toks:
                        newline += tok
                        if i == 0:
                            continue
                        elif ',' not in tok:
                            print(newline)
                            newline = ''

if __name__ == '__main__':
    main()

