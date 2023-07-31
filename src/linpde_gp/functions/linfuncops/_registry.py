from linpde_gp import functions, linfuncops

########################################################################################
# `SelectOutput` #######################################################################
########################################################################################


@linfuncops.SelectOutput.__call__.register  # pylint: disable=no-member
def _(self, f: functions.StackedFunction, /):
    return f.fns[self.idx]
