"""Adaptation of Atari environments for specification learning algorithms."""

import gym

from seals.util import AutoResetWrapper


def adventure_v5():
    """Fixed-length variant of Adventure-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Adventure-v5"))


def adventure_noframeskip():
    """Fixed-length variant of AdventureNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AdventureNoFrameskip-v4"))


def airraid_v5():
    """Fixed-length variant of AirRaid-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/AirRaid-v5"))


def airraid_noframeskip():
    """Fixed-length variant of AirRaidNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AirRaidNoFrameskip-v4"))


def alien_v5():
    """Fixed-length variant of Alien-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Alien-v5"))


def alien_noframeskip():
    """Fixed-length variant of AlienNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AlienNoFrameskip-v4"))


def amidar_v5():
    """Fixed-length variant of Amidar-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Amidar-v5"))


def amidar_noframeskip():
    """Fixed-length variant of AmidarNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AmidarNoFrameskip-v4"))


def assault_v5():
    """Fixed-length variant of Assault-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Assault-v5"))


def assault_noframeskip():
    """Fixed-length variant of AssaultNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AssaultNoFrameskip-v4"))


def asterix_v5():
    """Fixed-length variant of Asterix-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Asterix-v5"))


def asterix_noframeskip():
    """Fixed-length variant of AsterixNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AsterixNoFrameskip-v4"))


def asteroids_v5():
    """Fixed-length variant of Asteroids-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Asteroids-v5"))


def asteroids_noframeskip():
    """Fixed-length variant of AsteroidsNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AsteroidsNoFrameskip-v4"))


def atlantis_v5():
    """Fixed-length variant of Atlantis-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Atlantis-v5"))


def atlantis_noframeskip():
    """Fixed-length variant of AtlantisNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("AtlantisNoFrameskip-v4"))


def bankheist_v5():
    """Fixed-length variant of BankHeist-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/BankHeist-v5"))


def bankheist_noframeskip():
    """Fixed-length variant of BankHeistNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BankHeistNoFrameskip-v4"))


def battlezone_v5():
    """Fixed-length variant of BattleZone-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/BattleZone-v5"))


def battlezone_noframeskip():
    """Fixed-length variant of BattleZoneNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BattleZoneNoFrameskip-v4"))


def beamrider_v5():
    """Fixed-length variant of BeamRider-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/BeamRider-v5"))


def beamrider_noframeskip():
    """Fixed-length variant of BeamRiderNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BeamRiderNoFrameskip-v4"))


def berzerk_v5():
    """Fixed-length variant of Berzerk-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Berzerk-v5"))


def berzerk_noframeskip():
    """Fixed-length variant of BerzerkNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BerzerkNoFrameskip-v4"))


def bowling_v5():
    """Fixed-length variant of Bowling-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Bowling-v5"))


def bowling_noframeskip():
    """Fixed-length variant of BowlingNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BowlingNoFrameskip-v4"))


def boxing_v5():
    """Fixed-length variant of Boxing-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Boxing-v5"))


def boxing_noframeskip():
    """Fixed-length variant of BoxingNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BoxingNoFrameskip-v4"))


def breakout_v5():
    """Fixed-length variant of Breakout-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Breakout-v5"))


def breakout_noframeskip():
    """Fixed-length variant of BreakoutNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("BreakoutNoFrameskip-v4"))


def carnival_v5():
    """Fixed-length variant of Carnival-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Carnival-v5"))


def carnival_noframeskip():
    """Fixed-length variant of CarnivalNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("CarnivalNoFrameskip-v4"))


def centipede_v5():
    """Fixed-length variant of Centipede-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Centipede-v5"))


def centipede_noframeskip():
    """Fixed-length variant of CentipedeNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("CentipedeNoFrameskip-v4"))


def choppercommand_v5():
    """Fixed-length variant of ChopperCommand-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/ChopperCommand-v5"))


def choppercommand_noframeskip():
    """Fixed-length variant of ChopperCommandNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ChopperCommandNoFrameskip-v4"))


def crazyclimber_v5():
    """Fixed-length variant of CrazyClimber-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/CrazyClimber-v5"))


def crazyclimber_noframeskip():
    """Fixed-length variant of CrazyClimberNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("CrazyClimberNoFrameskip-v4"))


def defender_v5():
    """Fixed-length variant of Defender-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Defender-v5"))


def defender_noframeskip():
    """Fixed-length variant of DefenderNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("DefenderNoFrameskip-v4"))


def demonattack_v5():
    """Fixed-length variant of DemonAttack-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/DemonAttack-v5"))


def demonattack_noframeskip():
    """Fixed-length variant of DemonAttackNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("DemonAttackNoFrameskip-v4"))


def doubledunk_v5():
    """Fixed-length variant of DoubleDunk-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/DoubleDunk-v5"))


def doubledunk_noframeskip():
    """Fixed-length variant of DoubleDunkNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("DoubleDunkNoFrameskip-v4"))


def elevatoraction_v5():
    """Fixed-length variant of ElevatorAction-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/ElevatorAction-v5"))


def elevatoraction_noframeskip():
    """Fixed-length variant of ElevatorActionNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ElevatorActionNoFrameskip-v4"))


def enduro_v5():
    """Fixed-length variant of Enduro-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Enduro-v5"))


def enduro_noframeskip():
    """Fixed-length variant of EnduroNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("EnduroNoFrameskip-v4"))


def fishingderby_v5():
    """Fixed-length variant of FishingDerby-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/FishingDerby-v5"))


def fishingderby_noframeskip():
    """Fixed-length variant of FishingDerbyNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("FishingDerbyNoFrameskip-v4"))


def freeway_v5():
    """Fixed-length variant of Freeway-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Freeway-v5"))


def freeway_noframeskip():
    """Fixed-length variant of FreewayNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("FreewayNoFrameskip-v4"))


def frostbite_v5():
    """Fixed-length variant of Frostbite-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Frostbite-v5"))


def frostbite_noframeskip():
    """Fixed-length variant of FrostbiteNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("FrostbiteNoFrameskip-v4"))


def gopher_v5():
    """Fixed-length variant of Gopher-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Gopher-v5"))


def gopher_noframeskip():
    """Fixed-length variant of GopherNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("GopherNoFrameskip-v4"))


def gravitar_v5():
    """Fixed-length variant of Gravitar-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Gravitar-v5"))


def gravitar_noframeskip():
    """Fixed-length variant of GravitarNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("GravitarNoFrameskip-v4"))


def hero_v5():
    """Fixed-length variant of Hero-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Hero-v5"))


def hero_noframeskip():
    """Fixed-length variant of HeroNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("HeroNoFrameskip-v4"))


def icehockey_v5():
    """Fixed-length variant of IceHockey-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/IceHockey-v5"))


def icehockey_noframeskip():
    """Fixed-length variant of IceHockeyNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("IceHockeyNoFrameskip-v4"))


def jamesbond_v5():
    """Fixed-length variant of Jamesbond-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Jamesbond-v5"))


def jamesbond_noframeskip():
    """Fixed-length variant of JamesbondNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("JamesbondNoFrameskip-v4"))


def journeyescape_v5():
    """Fixed-length variant of JourneyEscape-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/JourneyEscape-v5"))


def journeyescape_noframeskip():
    """Fixed-length variant of JourneyEscapeNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("JourneyEscapeNoFrameskip-v4"))


def kangaroo_v5():
    """Fixed-length variant of Kangaroo-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Kangaroo-v5"))


def kangaroo_noframeskip():
    """Fixed-length variant of KangarooNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("KangarooNoFrameskip-v4"))


def krull_v5():
    """Fixed-length variant of Krull-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Krull-v5"))


def krull_noframeskip():
    """Fixed-length variant of KrullNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("KrullNoFrameskip-v4"))


def kungfumaster_v5():
    """Fixed-length variant of KungFuMaster-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/KungFuMaster-v5"))


def kungfumaster_noframeskip():
    """Fixed-length variant of KungFuMasterNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("KungFuMasterNoFrameskip-v4"))


def montezumarevenge_v5():
    """Fixed-length variant of MontezumaRevenge-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/MontezumaRevenge-v5"))


def montezumarevenge_noframeskip():
    """Fixed-length variant of MontezumaRevengeNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("MontezumaRevengeNoFrameskip-v4"))


def mspacman_v5():
    """Fixed-length variant of MsPacman-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/MsPacman-v5"))


def mspacman_noframeskip():
    """Fixed-length variant of MsPacmanNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("MsPacmanNoFrameskip-v4"))


def namethisgame_v5():
    """Fixed-length variant of NameThisGame-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/NameThisGame-v5"))


def namethisgame_noframeskip():
    """Fixed-length variant of NameThisGameNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("NameThisGameNoFrameskip-v4"))


def phoenix_v5():
    """Fixed-length variant of Phoenix-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Phoenix-v5"))


def phoenix_noframeskip():
    """Fixed-length variant of PhoenixNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("PhoenixNoFrameskip-v4"))


def pitfall_v5():
    """Fixed-length variant of Pitfall-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Pitfall-v5"))


def pitfall_noframeskip():
    """Fixed-length variant of PitfallNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("PitfallNoFrameskip-v4"))


def pong_v5():
    """Fixed-length variant of Pong-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Pong-v5"))


def pong_noframeskip():
    """Fixed-length variant of PongNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("PongNoFrameskip-v4"))


def pooyan_v5():
    """Fixed-length variant of Pooyan-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Pooyan-v5"))


def pooyan_noframeskip():
    """Fixed-length variant of PooyanNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("PooyanNoFrameskip-v4"))


def privateeye_v5():
    """Fixed-length variant of PrivateEye-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/PrivateEye-v5"))


def privateeye_noframeskip():
    """Fixed-length variant of PrivateEyeNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("PrivateEyeNoFrameskip-v4"))


def qbert_v5():
    """Fixed-length variant of Qbert-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Qbert-v5"))


def qbert_noframeskip():
    """Fixed-length variant of QbertNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("QbertNoFrameskip-v4"))


def riverraid_v5():
    """Fixed-length variant of Riverraid-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Riverraid-v5"))


def riverraid_noframeskip():
    """Fixed-length variant of RiverraidNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("RiverraidNoFrameskip-v4"))


def roadrunner_v5():
    """Fixed-length variant of RoadRunner-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/RoadRunner-v5"))


def roadrunner_noframeskip():
    """Fixed-length variant of RoadRunnerNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("RoadRunnerNoFrameskip-v4"))


def robotank_v5():
    """Fixed-length variant of Robotank-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Robotank-v5"))


def robotank_noframeskip():
    """Fixed-length variant of RobotankNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("RobotankNoFrameskip-v4"))


def seaquest_v5():
    """Fixed-length variant of Seaquest-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Seaquest-v5"))


def seaquest_noframeskip():
    """Fixed-length variant of SeaquestNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("SeaquestNoFrameskip-v4"))


def skiing_v5():
    """Fixed-length variant of Skiing-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Skiing-v5"))


def skiing_noframeskip():
    """Fixed-length variant of SkiingNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("SkiingNoFrameskip-v4"))


def solaris_v5():
    """Fixed-length variant of Solaris-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Solaris-v5"))


def solaris_noframeskip():
    """Fixed-length variant of SolarisNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("SolarisNoFrameskip-v4"))


def spaceinvaders_v5():
    """Fixed-length variant of SpaceInvaders-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/SpaceInvaders-v5"))


def spaceinvaders_noframeskip():
    """Fixed-length variant of SpaceInvadersNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("SpaceInvadersNoFrameskip-v4"))


def stargunner_v5():
    """Fixed-length variant of StarGunner-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/StarGunner-v5"))


def stargunner_noframeskip():
    """Fixed-length variant of StarGunnerNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("StarGunnerNoFrameskip-v4"))


def tennis_v5():
    """Fixed-length variant of Tennis-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Tennis-v5"))


def tennis_noframeskip():
    """Fixed-length variant of TennisNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("TennisNoFrameskip-v4"))


def timepilot_v5():
    """Fixed-length variant of TimePilot-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/TimePilot-v5"))


def timepilot_noframeskip():
    """Fixed-length variant of TimePilotNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("TimePilotNoFrameskip-v4"))


def tutankham_v5():
    """Fixed-length variant of Tutankham-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Tutankham-v5"))


def tutankham_noframeskip():
    """Fixed-length variant of TutankhamNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("TutankhamNoFrameskip-v4"))


def upndown_v5():
    """Fixed-length variant of UpNDown-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/UpNDown-v5"))


def upndown_noframeskip():
    """Fixed-length variant of UpNDownNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("UpNDownNoFrameskip-v4"))


def venture_v5():
    """Fixed-length variant of Venture-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Venture-v5"))


def venture_noframeskip():
    """Fixed-length variant of VentureNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("VentureNoFrameskip-v4"))


def videopinball_v5():
    """Fixed-length variant of VideoPinball-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/VideoPinball-v5"))


def videopinball_noframeskip():
    """Fixed-length variant of VideoPinballNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("VideoPinballNoFrameskip-v4"))


def wizardofwor_v5():
    """Fixed-length variant of WizardOfWor-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/WizardOfWor-v5"))


def wizardofwor_noframeskip():
    """Fixed-length variant of WizardOfWorNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("WizardOfWorNoFrameskip-v4"))


def yarsrevenge_v5():
    """Fixed-length variant of YarsRevenge-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/YarsRevenge-v5"))


def yarsrevenge_noframeskip():
    """Fixed-length variant of YarsRevengeNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("YarsRevengeNoFrameskip-v4"))


def zaxxon_v5():
    """Fixed-length variant of Zaxxon-v5.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ALE/Zaxxon-v5"))


def zaxxon_noframeskip():
    """Fixed-length variant of ZaxxonNoFrameskip-v4.

    When the environment would terminate, it instead resets. If wrapping with a further
    preprocessing wrapper, make sure to not terminate on life loss, since that would
    break the constant episode length property.
    """
    return AutoResetWrapper(gym.make("ZaxxonNoFrameskip-v4"))
