import data_tools as dt
import equation_systems as eqs
import matplotlib.pyplot as plt


def zadanieA(c, d, e, f):
    return dt.build_Ab(a1=5+e, a2=-1, a3=-1, f=f, N=900+10*c+d)


def zadanieB(A, b):
    _, jacobi_iterations_n, jacobi_res_norms, jacobi_prep_time, jacobi_iter_time =\
        eqs.jacobi_solve(A, b)
    _, gauss_seidl_iterations_n, gauss_seidl_res_norms, gauss_seidl_prep_time, gauss_seidl_iter_time =\
        eqs.gauss_seidl_solve(A, b)

    print('Zadanie B:')
    print(f'Metoda Jacobiego:'
          f'\n\tliczba iteracji: {jacobi_iterations_n},'
          f'\n\tczas przygotowywania macierzy: {round(jacobi_prep_time, 2)} s,'
          f'\n\tczas wykonywania iteracji: {round(jacobi_iter_time, 2)} s'
          f'\n\tostateczna norma residuum: {jacobi_res_norms[len(jacobi_res_norms) - 1]};')
    print(f'Metoda Gaussa-Seidla:'
          f'\n\tliczba iteracji: {gauss_seidl_iterations_n},'
          f'\n\tczas przygotowywania macierzy: {round(gauss_seidl_prep_time, 2)} s,'
          f'\n\tczas wykonywania iteracji: {round(gauss_seidl_iter_time, 2)} s'
          f'\n\tostateczna norma residuum: {gauss_seidl_res_norms[len(gauss_seidl_res_norms) - 1]};')
    print()

    plt.plot([i for i in range(jacobi_iterations_n + 1)], jacobi_res_norms, label='metoda Jacobiego')
    plt.plot([i for i in range(gauss_seidl_iterations_n + 1)], gauss_seidl_res_norms, label='metoda Gaussa-Seidla')
    plt.yscale('log')

    plt.title('Normy wektorów residuum w kolejnych iteracjach\nmetod Jacobiego i Gaussa-Seidla')
    plt.xlabel('numer iteracji')
    plt.ylabel('norma wektora residuum')
    plt.legend()
    plt.grid()

    plt.savefig('..\\report\\plots\\zadanieB.png')
    plt.close()


def zadanieC(c, d, f):
    A, b = dt.build_Ab(a1=3, a2=-1, a3=-1, f=f, N=900+10*c+d)

    _, jacobi_iterations_n, jacobi_res_norms, jacobi_prep_time, jacobi_iter_time =\
        eqs.jacobi_solve(A, b)
    _, gauss_seidl_iterations_n, gauss_seidl_res_norms, gauss_seidl_prep_time, gauss_seidl_iter_time =\
        eqs.gauss_seidl_solve(A, b)

    print('Zadanie C:')
    print(f'Metoda Jacobiego:'
          f'\n\tliczba iteracji: {jacobi_iterations_n},'
          f'\n\tczas przygotowywania macierzy: {round(jacobi_prep_time, 2)} s,'
          f'\n\tczas wykonywania iteracji: {round(jacobi_iter_time, 2)} s'
          f'\n\tostateczna norma residuum: {jacobi_res_norms[len(jacobi_res_norms) - 1]};')
    print(f'Metoda Gaussa-Seidla:'
          f'\n\tliczba iteracji: {gauss_seidl_iterations_n},'
          f'\n\tczas przygotowywania macierzy: {round(gauss_seidl_prep_time, 2)} s,'
          f'\n\tczas wykonywania iteracji: {round(gauss_seidl_iter_time, 2)} s'
          f'\n\tostateczna norma residuum: {gauss_seidl_res_norms[len(gauss_seidl_res_norms) - 1]};')
    print()

    plt.plot([i for i in range(jacobi_iterations_n + 1)], jacobi_res_norms, label='metoda Jacobiego')
    plt.plot([i for i in range(gauss_seidl_iterations_n + 1)], gauss_seidl_res_norms, label='metoda Gaussa-Seidla')
    plt.yscale('log')

    plt.title('Normy wektorów residuum w kolejnych iteracjach\nmetod Jacobiego i Gaussa-Seidla')
    plt.xlabel('numer iteracji')
    plt.ylabel('norma wektora residuum')
    plt.legend()
    plt.grid()

    plt.savefig('..\\report\\plots\\zadanieC.png')
    plt.close()


def zadanieD(c, d, f):
    A, b = dt.build_Ab(a1=3, a2=-1, a3=-1, f=f, N=900 + 10 * c + d)
    _, res_norm, fact_time, subst_time = eqs.LU_fact_solve(A, b)

    print('Zadanie D:')
    print(f'Metoda faktoryzacji LU:'
          f'\n\tczas faktoryzacji macierzy: {round(fact_time, 2)} s,'
          f'\n\tczas podstawiania (wprzód i wstecz): {round(subst_time, 2)} s'
          f'\n\tnorma residuum: {res_norm}')
    print()


def zadanieE(e, f):
    N = [100, 500, 1000, 2000, 3000]
    jacobi_times = []
    jacobi_iterations_numbers = []
    gauss_seidl_times = []
    gauss_seidl_iterations_numbers = []
    LU_fact_times = []

    print('Zadanie E:')

    for n in N:
        A, b = dt.build_Ab(a1=5+e, a2=-1, a3=-1, f=f, N=n)

        _, jacobi_iterations_n, jacobi_res_norms, jacobi_prep_time, jacobi_iter_time = \
            eqs.jacobi_solve(A, b)
        _, gauss_seidl_iterations_n, gauss_seidl_res_norms, gauss_seidl_prep_time, gauss_seidl_iter_time = \
            eqs.gauss_seidl_solve(A, b)
        _, LU_res_norm, LU_fact_fact_time, LU_fact_subst_time =\
            eqs.LU_fact_solve(A, b)

        jacobi_times.append(jacobi_prep_time + jacobi_iter_time)
        jacobi_iterations_numbers.append(jacobi_iterations_n)
        gauss_seidl_times.append(gauss_seidl_prep_time + gauss_seidl_iter_time)
        gauss_seidl_iterations_numbers.append(gauss_seidl_iterations_n)
        LU_fact_times.append(LU_fact_fact_time + LU_fact_subst_time)

        print(f'N = {n}:'
              f'\n\tmetoda Jacobiego: {round(jacobi_prep_time + jacobi_iter_time, 2)} s,'
              f'\n\tmetoda Gaussa-Seidla: {round(gauss_seidl_prep_time + gauss_seidl_iter_time, 2)} s,'
              f'\n\tmetoda faktoryzacji LU: {round(LU_fact_fact_time + LU_fact_subst_time, 2)} s')
    print()

    plt.plot(N, jacobi_times, label='metoda Jacobiego')
    plt.plot(N, gauss_seidl_times, label='metoda Gaussa-Seidla')
    plt.plot(N, LU_fact_times, label='metoda faktoryzacji LU')

    plt.title('Czas trwania poszczególnych algorytmów dla różnej liczby niewiadomych')
    plt.xlabel('liczba niewiadomych')
    plt.ylabel('czas trwania algorytmu [s]')
    plt.legend()
    plt.grid()

    plt.savefig('..\\report\\plots\\zadanieE_czas.png')
    plt.close()

    plt.plot(N, jacobi_iterations_numbers, label='metoda Jacobiego')
    plt.plot(N, gauss_seidl_iterations_numbers, label='metoda Gaussa-Seidla')

    plt.title('Liczba iteracji poszczególnych algorytmów dla różnej liczby niewiadomych')
    plt.xlabel('liczba niewiadomych')
    plt.ylabel('liczba iteracji')
    plt.legend()
    plt.grid()

    plt.savefig('..\\report\\plots\\zadanieE_iteracje.png')
    plt.close()
