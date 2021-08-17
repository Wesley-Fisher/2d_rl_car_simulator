import statistics

class Reporting:
    def __init__(self, settings):
        self.settings = settings

        self.car_performance = {}

    def record_car_performance(self, car_name, performance):
        if car_name not in self.car_performance:
            self.car_performance[car_name] = []
        self.car_performance[car_name].append(performance)

        if len(self.car_performance[car_name]) > self.settings.reporting.car_performance_length:
            self.car_performance[car_name].pop(0)

    def report_car_performance(self):
        for car in self.car_performance.keys():
            p = statistics.mean(self.car_performance[car])
            print("%s Performance:\t%f" % (car,p))

    def get_report(self):
        names = list(self.car_performance.keys())
        if len(names) == 0:
            return [],[]
        names.sort()
        values = [statistics.mean(self.car_performance[car]) for car in names]
        return names, values
