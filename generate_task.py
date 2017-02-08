# size = [1000, 5000, 10000, 15000, 20000, 25000, 30000]
size = [5000, 15000, 20000]
pars = [[8.0, 7.0], [2.0, 10.0], [0.6, 0.9], [3.0, 0.8], [4.0, 3.0]]
# noise = [0.05, 0.1, 0.2]
noise = [0.1, 0.2]

script = '$1'
output = '$2'
file = 'task.sh'

f = open(file, 'wb')

for s in size:
    for par in pars:
        command = 'python ' + script + \
                  ' -R 10 -size ' + str(s) + \
                  ' -A ' + str(par[0]) + ' -B ' + str(par[1]) + \
                  ' -o ' + output
                  # + ' -Hess'
                  # + ' -sample'

        command = 'echo ' + command + ' | submitAll.pl -l walltime=03:59:00'
        f.write(command)
        f.write('\n')

for s in size:
    for par in pars:
        for std in noise:
            command = 'python ' + script + \
                      ' -R 10 -size ' + str(s) + \
                      ' -A ' + str(par[0]) + ' -B ' + str(par[1]) + \
                      ' -o ' + output + ' -isNoise -std ' + str(std)
                      # + ' -Hess'
            command = 'echo ' + command + ' | submitAll.pl -l walltime=03:59:00'
            f.write(command)
            f.write('\n')

f.close()