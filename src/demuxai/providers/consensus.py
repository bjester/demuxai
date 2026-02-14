"""
TODO: an aggregating provider that queries multiple upstream providers
- asynchronously query multiple configured upstream providers
- uses one configured provider to generate the consensus between providers
- consensus provider model should probably have specific system prompt to NOT fact check--
    avoid bias towards its own information
- should expose upstream responses through Harmony format:
    <|channel|>analysis<|message|> [upstream responses] <|end|>
    <|channel|>final<|message|> [consensus response] <|return|>
"""
