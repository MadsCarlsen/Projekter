import numpy as np

# First rewrite the Mathematica file...
with open('Table_maker/Sph_harm_10.txt', 'r') as file:
    res_list = []
    for line in file.readlines():
        for string_i in line.split('\t'):
            string = string_i.replace('std::cos(theta)', 'z/r')
            # Fix that Mathematica does not convert numbers to floats..
            temp_string = ''
            old_char = 'a'
            in_num = False
            for char in string:
                if not char.isnumeric() and old_char.isnumeric() and char != '.' and not in_num:
                    temp_string += '.'
                if char == '.':
                    in_num = True
                elif not char.isnumeric():
                    in_num = False

                temp_string += char
                old_char = char
            string = temp_string

            # Find the exponent
            edit = string.split('std::')
            sign = ''
            if 'exp' in edit[1]:
                exp_val = edit[1]
                sign = '+'
            elif 'exp' in edit[-1]:
                exp_val = edit[-1]
                sign = '-'
            else:  # No exponent here!
                res_list.append(string.replace('\n', ''))
                continue

            # Get the exponent value
            exp_val = exp_val[4:]
            factor = ''
            while True:
                if exp_val[0] == '*' or exp_val[0] == 'I':
                    break
                factor += exp_val[0]
                exp_val = exp_val[1:]

            # Replace the values!
            if not factor:  # This means factor = 1
                res = string.replace('std::sin(theta)', '(x' + sign + 'I*y)/r')
                res = res.replace('std::exp(I*phi)', '1.')
            else:
                res = string.replace('std::pow(std::sin(theta),' + factor + ')',
                                     'std::pow((x' + sign + 'I*y)/r,' + factor + ')')
                res = res.replace('std::exp(' + factor + '*I*phi)', '1.')

            res_list.append(res.replace('\n', ''))


# Then write a new c++ file
max_l = int(np.sqrt(len(res_list)) - 1)

with open('sph_harm.h', 'w') as file:
    file.write('#include <cmath>\n')
    file.write('#include <complex>\n\n')
    file.write('using dcmplx = std::complex<double>;\n')
    file.write('constexpr dcmplx I(0., 1.);\n')
    file.write('constexpr double Pi = 3.14159265358979323846;\n\n')

    file.write('dcmplx sph_harm(dcmplx x, dcmplx y, dcmplx z, dcmplx r, int l, int m){\n')
    file.write('if(std::abs(m) > l){return 0.;}\n\n')

    file.write('if(l == 0){\n')
    file.write('return ' + res_list[0] + ';\n')
    file.write('}\n\n')

    counter = 1
    for l in range(1, max_l+1):
        file.write(f'else if (l == {l})' + '{\n')
        file.write(f'if(m == {-l})' + '{\n')
        file.write('return ' + res_list[counter] + ';\n')
        file.write('}\n')

        counter += 1
        for m in range(-l+1, l+1):
            file.write(f'else if (m == {m})' + '{\n')
            file.write('return ' + res_list[counter] + ';\n')
            file.write('}\n')
            if m == l:
                file.write('else{return 0.;}\n')
            counter += 1
        file.write('}\n\n')
    file.write('else{return 0.;}\n')
    file.write('}')

