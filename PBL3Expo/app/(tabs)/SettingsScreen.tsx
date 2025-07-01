import React, { useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TextInput,
  TouchableOpacity,
  Image,
  Switch,
  Alert,
} from 'react-native';
import CustomTabBar from '../../components/CustomTabBar';
import { User, Mail, Phone, MapPin, Lock, CheckCircle, Users, LogOut, Trash2 } from 'lucide-react-native';

const SettingsScreen = () => {
  // Account info state
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [phone, setPhone] = useState('');
  const [address, setAddress] = useState('');
  const [password, setPassword] = useState('');
  const [avatar, setAvatar] = useState('');

  // Notification toggles
  const [emailNotif, setEmailNotif] = useState(true);
  const [pushNotif, setPushNotif] = useState(true);
  const [inAppNotif, setInAppNotif] = useState(false);

  // Data consent
  const [dataConsent, setDataConsent] = useState(true);

  // Handlers (placeholders)
  const handleChangePhoto = () => Alert.alert('Change Photo', 'Photo picker not implemented.');
  const handleChangePassword = () => Alert.alert('Change Password', 'Password change not implemented.');
  const handleExportData = () => Alert.alert('Export Data', 'Data export not implemented.');
  const handleReportProblem = () => Alert.alert('Report a Problem', 'Support not implemented.');
  const handleReportIncorrect = () => Alert.alert('Report Incorrect Match', 'Support not implemented.');
  const handleFAQ = () => Alert.alert('FAQ/Help Center', 'Help center not implemented.');
  const handleLogout = () => Alert.alert('Log Out', 'Logout not implemented.');
  const handleDeleteAccount = () => Alert.alert('Delete Account', 'Account deletion not implemented.');
  const handleSave = () => Alert.alert('Save Changes', 'Settings saved (not really).');

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: '#f9fafb' }}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        {/* Account Information */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Information</Text>
          <View style={styles.avatarBox}>
            <View style={styles.avatarPlaceholder}>
              <User color="#facc15" size={48} />
            </View>
            <TouchableOpacity style={styles.changePhotoBtn} onPress={handleChangePhoto}>
              <Text style={styles.changePhotoText}>Change Photo</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.inputRow}><Text style={styles.label}>Full Name</Text><TextInput style={styles.input} value={fullName} onChangeText={setFullName} /></View>
          <View style={styles.inputRow}><Text style={styles.label}>Email Address</Text><TextInput style={styles.input} value={email} onChangeText={setEmail} keyboardType="email-address" autoCapitalize="none" /></View>
          <View style={styles.inputRow}><Text style={styles.label}>Phone Number</Text><TextInput style={styles.input} value={phone} onChangeText={setPhone} keyboardType="phone-pad" /></View>
          <View style={styles.inputRow}><Text style={styles.label}>Address/Region</Text><TextInput style={styles.input} value={address} onChangeText={setAddress} /></View>
          <TouchableOpacity style={styles.changePasswordBtn} onPress={handleChangePassword}>
            <Lock color="#facc15" size={16} style={{ marginRight: 8 }} />
            <Text style={styles.changePasswordText}>Change Password</Text>
          </TouchableOpacity>
          <View style={styles.roleRow}>
            <Text style={styles.roleText}>Role: <Text style={{ color: '#ef4444', fontWeight: 'bold' }}>Rescue Group</Text></Text>
            <Text style={styles.statusText}>Status: <Text style={{ color: '#22c55e', fontWeight: 'bold' }}>Verified</Text></Text>
          </View>
        </View>

        {/* Notifications */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Notifications</Text>
          <View style={styles.toggleRow}>
            <Text style={styles.toggleLabel}>Email Notifications</Text>
            <Switch value={emailNotif} onValueChange={setEmailNotif} trackColor={{ true: '#fde68a', false: '#e5e7eb' }} thumbColor={emailNotif ? '#facc15' : '#fff'} />
          </View>
          <View style={styles.toggleRow}>
            <Text style={styles.toggleLabel}>Push Notifications</Text>
            <Switch value={pushNotif} onValueChange={setPushNotif} trackColor={{ true: '#fde68a', false: '#e5e7eb' }} thumbColor={pushNotif ? '#facc15' : '#fff'} />
          </View>
          <View style={styles.toggleRow}>
            <Text style={styles.toggleLabel}>In-app Notifications</Text>
            <Switch value={inAppNotif} onValueChange={setInAppNotif} trackColor={{ true: '#fde68a', false: '#e5e7eb' }} thumbColor={inAppNotif ? '#facc15' : '#fff'} />
          </View>
        </View>

        {/* Privacy and Data */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Privacy and Data</Text>
          <TouchableOpacity style={styles.linkRow} onPress={() => Alert.alert('Privacy Policy', 'Privacy policy not implemented.')}> 
            <Text style={styles.linkText}>Privacy Policy</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.linkRow} onPress={() => Alert.alert('Terms and Services', 'Terms and services not implemented.')}> 
            <Text style={styles.linkText}>Terms and Services</Text>
          </TouchableOpacity>
          <View style={styles.toggleRow}>
            <Text style={styles.toggleLabel}>Data Usage Consent</Text>
            <Switch value={dataConsent} onValueChange={setDataConsent} trackColor={{ true: '#fde68a', false: '#e5e7eb' }} thumbColor={dataConsent ? '#facc15' : '#fff'} />
          </View>
          <TouchableOpacity style={styles.exportBtn} onPress={handleExportData}>
            <Text style={styles.exportBtnText}>Export My Data</Text>
          </TouchableOpacity>
        </View>

        {/* Support & Feedback */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Support & Feedback</Text>
          <TouchableOpacity style={styles.linkRow} onPress={handleReportProblem}>
            <Text style={styles.linkText}>Report a Problem</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.linkRow} onPress={handleReportIncorrect}>
            <Text style={styles.linkText}>Report an Incorrect Match</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.linkRow} onPress={handleFAQ}>
            <Text style={styles.linkText}>FAQ/Help Center</Text>
          </TouchableOpacity>
        </View>

        {/* Account Actions */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account Actions</Text>
          <TouchableOpacity style={styles.logoutBtn} onPress={handleLogout}>
            <LogOut color="#fff" size={18} style={{ marginRight: 8 }} />
            <Text style={styles.logoutText}>Log Out</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.deleteBtn} onPress={handleDeleteAccount}>
            <Trash2 color="#fff" size={18} style={{ marginRight: 8 }} />
            <Text style={styles.deleteText}>Delete Account</Text>
          </TouchableOpacity>
        </View>

        {/* Save Changes */}
        <TouchableOpacity style={styles.saveBtn} onPress={handleSave}>
          <Text style={styles.saveBtnText}>Save Changes</Text>
        </TouchableOpacity>
      </ScrollView>
      <CustomTabBar />
    </SafeAreaView>
  );
};

const styles = StyleSheet.create({
  scroll: { padding: 24, paddingBottom: 120 },
  section: { backgroundColor: '#fff', borderRadius: 16, padding: 20, marginBottom: 24, elevation: 1 },
  sectionTitle: { fontWeight: 'bold', fontSize: 16, color: '#222', marginBottom: 16 },
  avatarBox: { alignItems: 'center', marginBottom: 20 },
  avatar: { width: 80, height: 80, borderRadius: 40, backgroundColor: '#f3f4f6', marginBottom: 8 },
  avatarPlaceholder: {
    width: 80,
    height: 80,
    borderRadius: 40,
    backgroundColor: '#fffbe8',
    borderWidth: 2,
    borderColor: '#facc15',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 8,
  },
  changePhotoBtn: { backgroundColor: '#f3f4f6', borderRadius: 12, paddingVertical: 6, paddingHorizontal: 16 },
  changePhotoText: { color: '#facc15', fontWeight: 'bold', fontSize: 14 },
  inputRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12 },
  label: { width: 120, color: '#888', fontSize: 15 },
  input: { flex: 1, height: 40, backgroundColor: '#f3f4f6', borderRadius: 8, paddingHorizontal: 12, fontSize: 15 },
  changePasswordBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#f3f4f6', borderRadius: 12, paddingVertical: 8, paddingHorizontal: 16, alignSelf: 'flex-start', marginTop: 8, marginBottom: 8 },
  changePasswordText: { color: '#facc15', fontWeight: 'bold', fontSize: 14 },
  roleRow: { flexDirection: 'row', justifyContent: 'space-between', marginTop: 8 },
  roleText: { color: '#222', fontSize: 14 },
  statusText: { color: '#222', fontSize: 14 },
  toggleRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 },
  toggleLabel: { color: '#222', fontSize: 15 },
  linkRow: { paddingVertical: 10 },
  linkText: { color: '#2563eb', fontSize: 15 },
  exportBtn: { backgroundColor: '#f3f4f6', borderRadius: 12, paddingVertical: 12, alignItems: 'center', marginTop: 8 },
  exportBtnText: { color: '#222', fontWeight: 'bold', fontSize: 15 },
  logoutBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#f59e42', borderRadius: 12, paddingVertical: 12, justifyContent: 'center', marginBottom: 8 },
  logoutText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
  deleteBtn: { flexDirection: 'row', alignItems: 'center', backgroundColor: '#ef4444', borderRadius: 12, paddingVertical: 12, justifyContent: 'center' },
  deleteText: { color: '#fff', fontWeight: 'bold', fontSize: 15 },
  saveBtn: { backgroundColor: '#facc15', borderRadius: 20, paddingVertical: 18, alignItems: 'center', marginBottom: 32 },
  saveBtnText: { color: '#fff', fontWeight: 'bold', fontSize: 18 },
});

export default SettingsScreen; 