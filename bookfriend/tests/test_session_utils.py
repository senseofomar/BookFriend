from pathlib import Path

from bookfriend.utils import session_utils

TEST_PATH = "test_session.json"

def test_save_and_load_session():
    data ={"food":"bar", "count": 124}
    session_utils.save_session(data, TEST_PATH)

    assert Path(TEST_PATH).exists()    #old school way os.path.exists(TEST_PATH)

    loaded = session_utils.load_session(TEST_PATH)

    assert loaded == data

    Path(TEST_PATH).unlink()

def test_reset_session_removes_file(tmp_path):
    # Arrange: create a fake session file
    test_file = tmp_path / "session.json"
    test_file.write_text('{"foo": "bar"}', encoding="utf-8")
    assert test_file.exists()

    # Act: reset_session should delete it
    session_utils.reset_session(test_file)

    # Assert: file should not exist anymore
    assert not test_file.exists()

