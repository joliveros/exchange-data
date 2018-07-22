class Child(object):
    def __init__(self, *args, **kwds):
        if len(args) == 1 and str(type(args[0])) == "<class '__main__.Parent'>":
            new_args = [args[0].x, args[0].y, args[0].z]
            super(Child, self).__init__(*new_args, **kwds)
        else:
            super(Child, self).__init__(*args, **kwds)