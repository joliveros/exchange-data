import pprint

printer = pprint.PrettyPrinter(indent=2)


def nice_print(value):
    printer.pprint(value)
