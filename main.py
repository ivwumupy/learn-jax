import jax
import jax.numpy as np

global_list = []


def log2(x):
    global_list.append(x)
    ln_x = np.log(x)
    ln_2 = np.log(2.0)
    return ln_x / ln_2


def main():
    print(jax.make_jaxpr(log2))


if __name__ == "__main__":
    main()
