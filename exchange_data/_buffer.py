from io import StringIO


class Buffer(StringIO):

    def newline(self, count=1):
        n = 0
        while n <= count:
            self.write('\n')
            n += 1

    def section(self, value):
        self.write(f'### {value} ###')

    def __str__(self):
        return self.getvalue()
