from db import base as dbbase, repository as repo


def setup_module():
    dbbase.init_db()


def test_job_stores_options():
    with dbbase.get_session() as s:
        c = repo.create_case(s)
        f = repo.create_file(s, c, "a.wav", ".wav")
        j = repo.create_job(s, c, f, options={"separate": True})
        s.commit()
    with dbbase.get_session() as s:
        assert repo.get_job(s, j).options == {"separate": True}
