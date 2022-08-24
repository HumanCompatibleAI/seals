"""Adaptation of Atari environments for specification learning algorithms."""

import gym

from seals.util import AutoResetWrapper

# TODO(df) in documentation clarify that we need to use atari wrapper with terminal_on_life_loss=False


def adventure_v5():
    return AutoResetWrapper(gym.make("ALE/Adventure-v5"))


def adventure_noframeskip():
    return AutoResetWrapper(gym.make("AdventureNoFrameskip-v4"))


def airraid_v5():
    return AutoResetWrapper(gym.make("ALE/AirRaid-v5"))


def airraid_noframeskip():
    return AutoResetWrapper(gym.make("AirRaidNoFrameskip-v4"))


def alien_v5():
    return AutoResetWrapper(gym.make("ALE/Alien-v5"))


def alien_noframeskip():
    return AutoResetWrapper(gym.make("AlienNoFrameskip-v4"))


def amidar_v5():
    return AutoResetWrapper(gym.make("ALE/Amidar-v5"))


def amidar_noframeskip():
    return AutoResetWrapper(gym.make("AmidarNoFrameskip-v4"))


def assault_v5():
    return AutoResetWrapper(gym.make("ALE/Assault-v5"))


def assault_noframeskip():
    return AutoResetWrapper(gym.make("AssaultNoFrameskip-v4"))


def asterix_v5():
    return AutoResetWrapper(gym.make("ALE/Asterix-v5"))


def asterix_noframeskip():
    return AutoResetWrapper(gym.make("AsterixNoFrameskip-v4"))


def asteroids_v5():
    return AutoResetWrapper(gym.make("ALE/Asteroids-v5"))


def asteroids_noframeskip():
    return AutoResetWrapper(gym.make("AsteroidsNoFrameskip-v4"))


def atlantis_v5():
    return AutoResetWrapper(gym.make("ALE/Atlantis-v5"))


def atlantis_noframeskip():
    return AutoResetWrapper(gym.make("AtlantisNoFrameskip-v4"))


def bankheist_v5():
    return AutoResetWrapper(gym.make("ALE/BankHeist-v5"))


def bankheist_noframeskip():
    return AutoResetWrapper(gym.make("BankHeistNoFrameskip-v4"))


def battlezone_v5():
    return AutoResetWrapper(gym.make("ALE/BattleZone-v5"))


def battlezone_noframeskip():
    return AutoResetWrapper(gym.make("BattleZoneNoFrameskip-v4"))


def beamrider_v5():
    return AutoResetWrapper(gym.make("ALE/BeamRider-v5"))


def beamrider_noframeskip():
    return AutoResetWrapper(gym.make("BeamRiderNoFrameskip-v4"))


def berzerk_v5():
    return AutoResetWrapper(gym.make("ALE/Berzerk-v5"))


def berzerk_noframeskip():
    return AutoResetWrapper(gym.make("BerzerkNoFrameskip-v4"))


def bowling_v5():
    return AutoResetWrapper(gym.make("ALE/Bowling-v5"))


def bowling_noframeskip():
    return AutoResetWrapper(gym.make("BowlingNoFrameskip-v4"))


def boxing_v5():
    return AutoResetWrapper(gym.make("ALE/Boxing-v5"))


def boxing_noframeskip():
    return AutoResetWrapper(gym.make("BoxingNoFrameskip-v4"))


def breakout_v5():
    return AutoResetWrapper(gym.make("ALE/Breakout-v5"))


def breakout_noframeskip():
    return AutoResetWrapper(gym.make("BreakoutNoFrameskip-v4"))


def carnival_v5():
    return AutoResetWrapper(gym.make("ALE/Carnival-v5"))


def carnival_noframeskip():
    return AutoResetWrapper(gym.make("CarnivalNoFrameskip-v4"))


def centipede_v5():
    return AutoResetWrapper(gym.make("ALE/Centipede-v5"))


def centipede_noframeskip():
    return AutoResetWrapper(gym.make("CentipedeNoFrameskip-v4"))


def choppercommand_v5():
    return AutoResetWrapper(gym.make("ALE/ChopperCommand-v5"))


def choppercommand_noframeskip():
    return AutoResetWrapper(gym.make("ChopperCommandNoFrameskip-v4"))


def crazyclimber_v5():
    return AutoResetWrapper(gym.make("ALE/CrazyClimber-v5"))


def crazyclimber_noframeskip():
    return AutoResetWrapper(gym.make("CrazyClimberNoFrameskip-v4"))


def defender_v5():
    return AutoResetWrapper(gym.make("ALE/Defender-v5"))


def defender_noframeskip():
    return AutoResetWrapper(gym.make("DefenderNoFrameskip-v4"))


def demonattack_v5():
    return AutoResetWrapper(gym.make("ALE/DemonAttack-v5"))


def demonattack_noframeskip():
    return AutoResetWrapper(gym.make("DemonAttackNoFrameskip-v4"))


def doubledunk_v5():
    return AutoResetWrapper(gym.make("ALE/DoubleDunk-v5"))


def doubledunk_noframeskip():
    return AutoResetWrapper(gym.make("DoubleDunkNoFrameskip-v4"))


def elevatoraction_v5():
    return AutoResetWrapper(gym.make("ALE/ElevatorAction-v5"))


def elevatoraction_noframeskip():
    return AutoResetWrapper(gym.make("ElevatorActionNoFrameskip-v4"))


def enduro_v5():
    return AutoResetWrapper(gym.make("ALE/Enduro-v5"))


def enduro_noframeskip():
    return AutoResetWrapper(gym.make("EnduroNoFrameskip-v4"))


def fishingderby_v5():
    return AutoResetWrapper(gym.make("ALE/FishingDerby-v5"))


def fishingderby_noframeskip():
    return AutoResetWrapper(gym.make("FishingDerbyNoFrameskip-v4"))


def freeway_v5():
    return AutoResetWrapper(gym.make("ALE/Freeway-v5"))


def freeway_noframeskip():
    return AutoResetWrapper(gym.make("FreewayNoFrameskip-v4"))


def frostbite_v5():
    return AutoResetWrapper(gym.make("ALE/Frostbite-v5"))


def frostbite_noframeskip():
    return AutoResetWrapper(gym.make("FrostbiteNoFrameskip-v4"))


def gopher_v5():
    return AutoResetWrapper(gym.make("ALE/Gopher-v5"))


def gopher_noframeskip():
    return AutoResetWrapper(gym.make("GopherNoFrameskip-v4"))


def gravitar_v5():
    return AutoResetWrapper(gym.make("ALE/Gravitar-v5"))


def gravitar_noframeskip():
    return AutoResetWrapper(gym.make("GravitarNoFrameskip-v4"))


def hero_v5():
    return AutoResetWrapper(gym.make("ALE/Hero-v5"))


def hero_noframeskip():
    return AutoResetWrapper(gym.make("HeroNoFrameskip-v4"))


def icehockey_v5():
    return AutoResetWrapper(gym.make("ALE/IceHockey-v5"))


def icehockey_noframeskip():
    return AutoResetWrapper(gym.make("IceHockeyNoFrameskip-v4"))


def jamesbond_v5():
    return AutoResetWrapper(gym.make("ALE/Jamesbond-v5"))


def jamesbond_noframeskip():
    return AutoResetWrapper(gym.make("JamesbondNoFrameskip-v4"))


def journeyescape_v5():
    return AutoResetWrapper(gym.make("ALE/JourneyEscape-v5"))


def journeyescape_noframeskip():
    return AutoResetWrapper(gym.make("JourneyEscapeNoFrameskip-v4"))


def kangaroo_v5():
    return AutoResetWrapper(gym.make("ALE/Kangaroo-v5"))


def kangaroo_noframeskip():
    return AutoResetWrapper(gym.make("KangarooNoFrameskip-v4"))


def krull_v5():
    return AutoResetWrapper(gym.make("ALE/Krull-v5"))


def krull_noframeskip():
    return AutoResetWrapper(gym.make("KrullNoFrameskip-v4"))


def kungfumaster_v5():
    return AutoResetWrapper(gym.make("ALE/KungFuMaster-v5"))


def kungfumaster_noframeskip():
    return AutoResetWrapper(gym.make("KungFuMasterNoFrameskip-v4"))


def montezumarevenge_v5():
    return AutoResetWrapper(gym.make("ALE/MontezumaRevenge-v5"))


def montezumarevenge_noframeskip():
    return AutoResetWrapper(gym.make("MontezumaRevengeNoFrameskip-v4"))


def mspacman_v5():
    return AutoResetWrapper(gym.make("ALE/MsPacman-v5"))


def mspacman_noframeskip():
    return AutoResetWrapper(gym.make("MsPacmanNoFrameskip-v4"))


def namethisgame_v5():
    return AutoResetWrapper(gym.make("ALE/NameThisGame-v5"))


def namethisgame_noframeskip():
    return AutoResetWrapper(gym.make("NameThisGameNoFrameskip-v4"))


def phoenix_v5():
    return AutoResetWrapper(gym.make("ALE/Phoenix-v5"))


def phoenix_noframeskip():
    return AutoResetWrapper(gym.make("PhoenixNoFrameskip-v4"))


def pitfall_v5():
    return AutoResetWrapper(gym.make("ALE/Pitfall-v5"))


def pitfall_noframeskip():
    return AutoResetWrapper(gym.make("PitfallNoFrameskip-v4"))


def pong_v5():
    return AutoResetWrapper(gym.make("ALE/Pong-v5"))


def pong_noframeskip():
    return AutoResetWrapper(gym.make("PongNoFrameskip-v4"))


def pooyan_v5():
    return AutoResetWrapper(gym.make("ALE/Pooyan-v5"))


def pooyan_noframeskip():
    return AutoResetWrapper(gym.make("PooyanNoFrameskip-v4"))


def privateeye_v5():
    return AutoResetWrapper(gym.make("ALE/PrivateEye-v5"))


def privateeye_noframeskip():
    return AutoResetWrapper(gym.make("PrivateEyeNoFrameskip-v4"))


def qbert_v5():
    return AutoResetWrapper(gym.make("ALE/Qbert-v5"))


def qbert_noframeskip():
    return AutoResetWrapper(gym.make("QbertNoFrameskip-v4"))


def riverraid_v5():
    return AutoResetWrapper(gym.make("ALE/Riverraid-v5"))


def riverraid_noframeskip():
    return AutoResetWrapper(gym.make("RiverraidNoFrameskip-v4"))


def roadrunner_v5():
    return AutoResetWrapper(gym.make("ALE/RoadRunner-v5"))


def roadrunner_noframeskip():
    return AutoResetWrapper(gym.make("RoadRunnerNoFrameskip-v4"))


def robotank_v5():
    return AutoResetWrapper(gym.make("ALE/Robotank-v5"))


def robotank_noframeskip():
    return AutoResetWrapper(gym.make("RobotankNoFrameskip-v4"))


def seaquest_v5():
    return AutoResetWrapper(gym.make("ALE/Seaquest-v5"))


def seaquest_noframeskip():
    return AutoResetWrapper(gym.make("SeaquestNoFrameskip-v4"))


def skiing_v5():
    return AutoResetWrapper(gym.make("ALE/Skiing-v5"))


def skiing_noframeskip():
    return AutoResetWrapper(gym.make("SkiingNoFrameskip-v4"))


def solaris_v5():
    return AutoResetWrapper(gym.make("ALE/Solaris-v5"))


def solaris_noframeskip():
    return AutoResetWrapper(gym.make("SolarisNoFrameskip-v4"))


def spaceinvaders_v5():
    return AutoResetWrapper(gym.make("ALE/SpaceInvaders-v5"))


def spaceinvaders_noframeskip():
    return AutoResetWrapper(gym.make("SpaceInvadersNoFrameskip-v4"))


def stargunner_v5():
    return AutoResetWrapper(gym.make("ALE/StarGunner-v5"))


def stargunner_noframeskip():
    return AutoResetWrapper(gym.make("StarGunnerNoFrameskip-v4"))


def tennis_v5():
    return AutoResetWrapper(gym.make("ALE/Tennis-v5"))


def tennis_noframeskip():
    return AutoResetWrapper(gym.make("TennisNoFrameskip-v4"))


def timepilot_v5():
    return AutoResetWrapper(gym.make("ALE/TimePilot-v5"))


def timepilot_noframeskip():
    return AutoResetWrapper(gym.make("TimePilotNoFrameskip-v4"))


def tutankham_v5():
    return AutoResetWrapper(gym.make("ALE/Tutankham-v5"))


def tutankham_noframeskip():
    return AutoResetWrapper(gym.make("TutankhamNoFrameskip-v4"))


def upndown_v5():
    return AutoResetWrapper(gym.make("ALE/UpNDown-v5"))


def upndown_noframeskip():
    return AutoResetWrapper(gym.make("UpNDownNoFrameskip-v4"))


def venture_v5():
    return AutoResetWrapper(gym.make("ALE/Venture-v5"))


def venture_noframeskip():
    return AutoResetWrapper(gym.make("VentureNoFrameskip-v4"))


def videopinball_v5():
    return AutoResetWrapper(gym.make("ALE/VideoPinball-v5"))


def videopinball_noframeskip():
    return AutoResetWrapper(gym.make("VideoPinballNoFrameskip-v4"))


def wizardofwor_v5():
    return AutoResetWrapper(gym.make("ALE/WizardOfWor-v5"))


def wizardofwor_noframeskip():
    return AutoResetWrapper(gym.make("WizardOfWorNoFrameskip-v4"))


def yarsrevenge_v5():
    return AutoResetWrapper(gym.make("ALE/YarsRevenge-v5"))


def yarsrevenge_noframeskip():
    return AutoResetWrapper(gym.make("YarsRevengeNoFrameskip-v4"))


def zaxxon_v5():
    return AutoResetWrapper(gym.make("ALE/Zaxxon-v5"))


def zaxxon_noframeskip():
    return AutoResetWrapper(gym.make("ZaxxonNoFrameskip-v4"))
