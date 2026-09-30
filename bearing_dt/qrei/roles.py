"""Fold-role allocations for the leave-one-bearing-out design."""


ROLE_OFFSETS={"forward":(1,2),"swap":(2,1),"reverse":(-1,-2)}


def fold_roles(bearings,test,roles="forward"):
    """Primary roles are 'forward'; 'swap' and 'reverse' are the registered sensitivity allocations."""
    k=bearings.index(test)
    v,c=ROLE_OFFSETS[roles]
    validation=bearings[(k+v)%len(bearings)]
    calibration=bearings[(k+c)%len(bearings)]
    train=[b for b in bearings if b not in (test,validation,calibration)]
    return train,validation,calibration
