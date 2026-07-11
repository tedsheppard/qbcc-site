# Firm Precedent Vault — isolated feature package.
#
# Nothing in here is imported by existing live code paths except the two
# wiring points in server.py (router include + scheduler jobs), both of
# which are wrapped so a failure here can never take the main site down.
