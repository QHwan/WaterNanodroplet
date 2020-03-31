from __future__ import print_function, division, absolute_import

class Parameter(object):
    """Parameters used in molecular dynamics simulation"""

    def __init__(self):
        self._water_model = 'TIP4P/2005'
        self._atom_type_vec = ['OW', 'HW1', 'HW2', 'MW']
        # atomic mass
        self._mass_dict = {'OW': 15.999,
                           'HW1': 1.008,
                           'HW2': 1.008,
                           'MW': 0.000}
        # atomic charge: unit (e)
        self._charge_dict = {'OW': 0.0000,
                            'HW1': 0.5564,
                            'HW2': 0.5564,
                            'MW': -1.1128}
        # LJ sigma: unit (nm)
        self._sigma_dict = {'OW': 0.31589,
                           'HW1': 0.,
                           'HW2': 0.,
                           'MW': 0.}
        # LJ epsilon: unit (kJ/mol)
        self._epsilon_dict = {'OW': 0.7749,
                             'HW1': 0.,
                             'HW2': 0.,
                             'MW': 0}

        # check missing parameters
        for atom_type in self._atom_type_vec:
            if atom_type not in self._mass_dict:
                raise KeyError('You mass atom mass.')
            if atom_type not in self._charge_dict:
                raise KeyError('You miss charge of atom.')
            if atom_type not in self._sigma_dict:
                raise KeyError('You miss LJ sigma parameter of atom.')
            if atom_type not in self._epsilon_dict:
                raise KeyError('You miss LJ epsilon parameter of atom.')

    @property
    def water_model(self):
        return self._water_model

    @property
    def mass_dict(self):
        return self._mass_dict

    @property
    def charge_dict(self):
        return self._charge_dict

    @property
    def sigma_dict(self):
        return self._sigma_dict

    @property
    def epsilon_dict(self):
        return self._epsilon_dict


