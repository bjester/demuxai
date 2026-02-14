class RegistryOverwriteError(Exception):
    pass


class UnregisteredError(Exception):
    pass


class ProviderNotFoundError(Exception):
    pass


class ProviderConfigurationError(Exception):
    pass
