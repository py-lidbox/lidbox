* rewrite pipeline to have consistent state boundaries and robust commands for state transitions
* be more consistent with the datagroups, specify datagroup keys for all actions, e.g. do not extract features for 'test' datagroup
* split commands into package by state, make sure external dependencies are not imported when not needed
* move generalizable logic away from cli interface for providing notebook api
