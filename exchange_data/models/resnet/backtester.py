import alog
import click


@click.command()
# @click.option('--epochs', '-e', type=int, default=10)
# @click.option('--batch-size', '-b', type=int, default=20)
# @click.option('--learning-rate', '-l', type=float, default=0.3e-4)
# @click.option('--clear', '-c', is_flag=True)
# @click.option('--eval-span', type=str, default='20m')
def main():
    alog.info('## go ##')


if __name__ == '__main__':
    main()
